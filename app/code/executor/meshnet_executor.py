import torch
import os
import random  
import numpy as np  
import nibabel as nib
import json
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from .meshnet import MeshNet  # Import the MeshNet model
from .meshnet import enMesh_checkpoint
from .loader import Scanloader  # Import the Scanloader for MRI data
from .dist import GenericLogger  # Import GenericLogger
from .dice import faster_dice  # Import Dice score calculation
import torch.cuda.amp as amp
from torch.utils.checkpoint import checkpoint  # For layer checkpointing
from .paths import get_data_directory_path, get_output_directory_path
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

class MeshNetExecutor(Executor):
    def __init__(self):
        super().__init__()
        # Model Initialization
        config_file_path = os.path.join(os.path.dirname(__file__), "modelAE.json")

        # Set a fixed random seed for reproducibility across all sites
        # By testing found seed method causes memory issue <<<<
        # Ensure consistent initialization of weights
        # Practically, we can initialize the model on one site and broadcast the initial weights to all other sites.
        # self.set_seed(42)  # The seed value can be changed as per your preference     


        # Logger can be found for example with: MeshDist_nvflare/simulator_workspace/simulate_job/app_site-1 and app_site-2
        self.logger = GenericLogger(log_file_path='meshnet_executor.log')


        self.model = enMesh_checkpoint(in_channels=1, n_classes=3, channels=5, config_file=config_file_path)

        # Check if GPU available
        # GPU assignment
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set GPU ID explicitly
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # Set the environment variable PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to help PyTorch handle fragmented memory
        # This env var helps in avoid cuda out of memory message. MeshNetExecutor: OutOfMemoryError: CUDA out of memory.

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


        # # Check if the class variable initial_weights is None  ( Failded becasue both sites give copy)
        # if MeshNetExecutor.initial_weights is None:
        #     # If this is the first site, store the model's initial weights
        #     MeshNetExecutor.initial_weights = {key: value.clone() for key, value in self.model.state_dict().items()}
        #     self.logger.log_message(f"Initial weights copied from this site")
        # else:
        #     # If weights have already been initialized, load them into this site's model
        #     self.model.load_state_dict(MeshNetExecutor.initial_weights)
        #     self.logger.log_message(f"Initial weights assigned to this site.")


        # Define the file path for the weights   
        initial_weights_file_path = os.path.join(os.path.dirname(__file__), "initial_weights.pth") 
        

        # Check if the initial weights file exists
        if os.path.exists(initial_weights_file_path):
            # If weights have already been saved, load them from the file
            # self.model.load_state_dict(torch.load(initial_weights_file_path))
            self.model.load_state_dict(torch.load(initial_weights_file_path, weights_only=True))
            self.logger.log_message(f"Loaded initial weights from file for this site.")
        else:
            # Save initial weights to file (first site)
            torch.save(self.model.state_dict(), initial_weights_file_path)
            self.logger.log_message(f"Initial weights saved to file for this site.")        


        # Ensure model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True

        # Initialize the variable to store the previous learning rate (initially None)
        self.learning_rate =  0.0005    #<<<<<<<<<<<<<<<<<<<<<<<<

        self.previous_lr = 0

        # Optimizer and criterion setup
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) 




        # Criterion: CrossEntropy for loss calculation with class weights and label smoothing
        class_weight = torch.FloatTensor([0.2, 1.0, 0.8]).to(self.device)  # Adjust weights based on class balance
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.01)


        # Add learning rate scheduler ( Overlook for now)     <<<<<<<<<<
        # This will reduce the learning rate by a factor of 0.1 if the validation loss does not improve for 5 epochs.
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        
        self.current_epoch = 0

        # Epochs and aggregation interval
        self.total_epochs = 10  # Set the total number of epochs          #<<<<<<<<<<<<<<<<<<<<<<<<

        self.target_dice = 0.95 # learning stop once reach this dice
        self.stop_training = False  # Flag to control breaking out of the outer loop

        # self.aggregation_interval = 1  # Aggregation occurs every 5 epochs (you can modify this)

        # self.dice_threshold = 0.9  # Set the Dice score threshold

        # Gradient accumulation  ( Overlook for now)  <<<<<<<<<<
        # self.gradient_accumulation_steps = 4

        # Early stopping   ( Overlook for now)  <<<<<<<<<<
        # self.early_stopping_patience = 10
        # self.best_loss = float('inf')
        # self.epochs_without_improvement = 0

        # Initialization flag for data loader
        self.data_loader_initialized = False

        self.log_image_label_shapes_once = True  # Flag to log tensor shapes only once

        self.log_shapes_once = True  # Flag to log tensor shapes only once

        self.nifti_saved = False  # Flag to ensure we save only one sample

        self.dice_threshold_to_save_output = 0.3   # save output sample when dice above the threshold # <<<<<<<<<<<<<<<<<<<<<<<<


        self.shuffle_training = True  # Set to True if shuffling is desired for standalone training
        self.use_split_file = True




    # def set_seed(self, seed):
    #     # Set the random seed for reproducibility
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed(seed)
    #         torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    #     torch.backends.cudnn.benchmark = False

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        
        # Get site name
        self.site_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)


        # Check for initial model broadcast 
        # (((( FOR FUTUR USE))))  as practicall solution to share init weights between sites at begining. 
        # if task_name == "initialize_model":
        #     self.logger.log_message(f"{self.site_name}-initialize_model weights called")
        #     model_weights = shareable["model_weights"]
        #     self.load_model_weights(model_weights)
        #     return Shareable()

        # if "initial_weights" in shareable:
        #     self.logger.log_message(f"{self.site_name} - Received initial weights from the server.")
        #     initial_weights = shareable["initial_weights"]
        #     self.model.load_state_dict(initial_weights)  # Load the initial weights
        #     return Shareable()  # Acknowledge receipt        


        # Initialize data loader and other site-specific configurations once
        if not self.data_loader_initialized:
            # Get the correct data directory path
            db_file_path = os.path.join(get_data_directory_path(fl_ctx), "mindboggle.db")
            split_file_path = os.path.join(os.path.dirname(__file__), "splits.json")

            # Initialize Data Loader with dynamic path
            self.data_loader = Scanloader(db_file=db_file_path, label_type='GWlabels', num_cubes=1, 
                                          use_split_file=self.use_split_file, split_file=split_file_path, subset="train", logger=self.logger)
            self.trainloader, self.validloader, self.testloader = self.data_loader.get_loaders(batch_size=1, shuffle=self.shuffle_training)


            self.shape = 256
            self.current_iteration = 0

            self.train_size = len(self.trainloader)
            self.iterations = self.total_epochs * self.train_size

            # Initialize the scheduler now that trainloader is available
            # A very high learning rate essentially forces the model to discard previous learning and start afresh, disrupting convergence.
            # Gradually warm up the learning rate instead of allowing it to spike early
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=0.005, # Peak learning rate, the lower the less likley to overshoot during training
                anneal_strategy='linear',  # Gradual increase and decrease
                pct_start=0.1,  # Slow warm-up over the first 10% of training
                total_steps=self.iterations if self.iterations > 0 else 1   # Exact number of steps
                # epochs=self.total_epochs,
                # steps_per_epoch=len(self.trainloader) if len(self.trainloader) > 0 else 1
            )


            # Set flag to True so the loader is only initialized once
            self.data_loader_initialized = True
            self.logger.log_message(f"Data loader initialized for site {self.site_name}")
            self.logger.log_message(f"Train size :{self.train_size} samples, Valid size: {len(self.validloader)} samples")
            self.logger.log_message(f"Total epochs: {self.total_epochs}, Total iterations: {self.iterations}, lr: {self.learning_rate}")


        if task_name == "train_and_get_gradients":
            self.logger.log_message(f"{self.site_name}-train_and_get_gradients called ")
            gradients = self.train_and_get_gradients(fl_ctx)
            outgoing_shareable = Shareable()
            outgoing_shareable["gradients"] = gradients
            return outgoing_shareable

        elif task_name == "accept_aggregated_gradients":
            self.logger.log_message(f"{self.site_name}-accept_aggregated_gradients called ")
            aggregated_gradients = shareable["aggregated_gradients"]
            self.apply_gradients(aggregated_gradients, fl_ctx)
            return Shareable()



    # def train_and_get_gradients_old(self, fl_ctx):
    #     # Set the model to training mode, so dropout is activated
    #     # and BatchNorm normalizes the input based on the statistics of the current mini-batch.
    #     self.model.train()

    #     # Initialize accumulators for the loss and gradients
    #     total_loss = 0.0

    

    #     # Training loop for one epoch (full pass through the dataset)

    #     for batch_id, (image, label) in enumerate(self.trainloader):

    #         # Moving data to the correct device (CPU or GPU)
    #         image, label = image.to(self.device), label.to(self.device)

    #         self.optimizer.zero_grad() #  It resets/clears the gradients of all model parameters (i.e., weights).
    #         # PyTorch by defualt adds new gradients to any existing gradients, to prevent this accumulation of gradients 
    #         # from previous iterations, manually set them to zero at the beginning of each new training iteration.


    #         # Mixed precision and checkpointing
    #         with torch.amp.autocast(device_type='cuda'):
    #             # Forward passing input data through the model to get predictions, start training
                
    #             # reshape the 3D image tensor into a 5D tensor with the shape [batch_size, channels, depth, height, width].
    #             # -1: The batch size dimension is inferred.
    #             #  1: This indicates a single channel (grayscale MRI image).
    #             output = torch.utils.checkpoint.checkpoint(self.model, image.reshape(-1, 1, self.shape, self.shape, self.shape), use_reentrant=False)

    #             labels = torch.squeeze(label)  # Squeeze the label
    #             # Training label shape: torch.Size([1, 256, 256, 256])
    #             # Squeeze labels shape : [ 256, 256, 256]

    #             labels = (labels * 2).round().long()  # Multiply by 2, round the values, and cast to long


    #             # Log the shapes and unique values of the image and label once
    #             if self.log_image_label_shapes_once:
    #                 # Log image and label shapes
    #                 self.logger.log_message(f"Training image shape: {image.shape}")
    #                 # Training image shape: torch.Size([1, 256, 256, 256])

    #                 self.logger.log_message(f"Training label shape: {label.shape}")
    #                 # Training label shape: torch.Size([1, 256, 256, 256])

    #                 self.logger.log_message(f"Training output shape: {output.shape}")
    #                 #  Training output shape: torch.Size([1, 3, 256, 256, 256])
                    
    #                 # Log the unique values in both image and label
    #                 unique_label = torch.unique(label)
    #                 unique_labels = torch.unique(labels)
    #                 self.logger.log_message(f"Unique values in training GT label: {unique_label.tolist()}")
    #                 #  Unique values in training label: [0.0, 0.5, 1.0]    <<<<<<<<<<<<<<<<< Normalized by Pratyush
                    
    #                 self.logger.log_message(f"Unique values in training sequeezed long scaled labels: {unique_labels.tolist()}")
    #                 #  Unique values in training label: [0, 1, 2]   

    #                 self.log_image_label_shapes_once = False  # Set to False so this is only logged once                

    #             # compute the loss between the predicted output and the ground truth labels
    #             # The label tensor is reshaped to [batch_size, depth, height, width] to match the output shape of the model.
    #             # .long() * 2: Double the label values. Looks like Pratyush made it 0, 0.5, 1 range. 

    #             #  For CrossEntropyLoss, the input (predictions) should have 
    #             #  shape [batch_size, num_classes, height, width, depth] (as the output does),
    #             #  while the target (label) should have shape [batch_size, height, width, depth] containing class indices as integers.                



    #             loss = self.criterion(output, labels.reshape(-1, self.shape, self.shape, self.shape))


    #         # Accumulate loss
    #         total_loss += loss.item()

    #         # Scale loss and backward pass
    #         # self.scaler.scale(loss).backward()

    #         # Scale loss and backward pass (Backpropagation)
    #         # Backward pass (gradient calculation): Calculate the gradients for all the model's parameters with respect to the loss:
    #         loss.backward() #  calculate gradients

    #         self.optimizer.step() # gradients are applied to the model parameters 
    #         # Training done for that round.   

    #         # Update the learning rate 
    #         self.scheduler.step()

    #         # Get the current learning rate
    #         current_lr = self.optimizer.param_groups[0]['lr']

    #         # Check if the learning rate has changed, and log it if so
    #         if current_lr != self.previous_lr:
    #             self.logger.log_message(f"Learning rate changed from {self.previous_lr} to: {current_lr}")
    #             self.previous_lr = current_lr  # Update the previous learning rate


    #         # # Accumulate gradients without updating yet
    #         # if (batch_id + 1) % self.gradient_accumulation_steps == 0:
    #         #     # Update optimizer
    #         #     self.scaler.step(self.optimizer)
    #         #     self.scaler.update()
    #         #     self.optimizer.zero_grad()


    #     # Clear GPU cache (No need)
    #     # torch.cuda.empty_cache()

    #     # Log the average loss and Dice score per epoch
    #     average_loss = total_loss / len(self.trainloader)

    #     # dice_score = self.calculate_dice(self.trainloader)
    #     # Calculate Dice score on the validation set
    #     # self.model.eval()  # Set the model to evaluation mode for validation
    #     dice_score = self.calculate_dice(self.validloader, fl_ctx)  # Use validation set        
    #     self.logger.log_message(f"{self.site_name} - Epoch {self.current_epoch}: Loss = {average_loss}, Val Dice = {dice_score}")



    #     # Check for early stopping           <<<<<<<<<<<<<<<<<<  (Overlooked for now)
    #     # if average_loss < self.best_loss:
    #     #     self.best_loss = average_loss
    #     #     self.epochs_without_improvement = 0
    #     # else:
    #     #     self.epochs_without_improvement += 1

    #     # if self.epochs_without_improvement >= self.early_stopping_patience:
    #     #     self.logger.log_message(f"Early stopping triggered at epoch {self.current_epoch}")
    #     #     return []

    #     self.logger.log_message(f"{self.site_name} Preparing payload after an iteration in epoch {self.current_epoch}")
    #     # return [grad.clone().cpu().numpy() for grad in gradient_accumulator if grad is not None]

    #     # Accumulate gradients
    #     gradients = []
    #     for i, param in enumerate(self.model.parameters()):
    #         if param.grad is not None:
    #             gradients.append(param.grad.clone().cpu().numpy())
        

    #     # for example :
    #     # gradients = [
    #     #     array([[ 0.01, -0.02], [ 0.03,  0.04]]),  # gradient for some weight matrix (e.g shape [2, 2])
    #     #     array([0.005, -0.015]),                   # gradient for some bias vector (e.g shape [2])
    #     #     array([[ 0.02, 0.01], [-0.03, 0.05]])     # Another gradient for a different weight matrix and so on.
    #     # ]



    #     return gradients        


    # Define functions to save and load checkpoints
    def save_checkpoint(model, optimizer, scheduler, epoch, filename="checkpoint.pth"):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, filename)
        print(f"Checkpoint saved at epoch {epoch}")


    def train_and_get_gradients(self, fl_ctx):
        # Set the model to training mode, so dropout is activated
        # and BatchNorm normalizes the input based on the statistics of the current mini-batch.
        self.model.train()

        # Initialize accumulators for the loss and gradients
        total_loss = 0.0
        total_train_dice = 0.0

    
        # Training loop for one epoch (full pass through the dataset)

        for batch_id, (image, label) in enumerate(self.trainloader):

            # Moving data to the correct device (CPU or GPU)
            image, label = image.to(self.device), label.to(self.device)

            self.optimizer.zero_grad() #  It resets/clears the gradients of all model parameters (i.e., weights).
            # PyTorch by defualt adds new gradients to any existing gradients, to prevent this accumulation of gradients 
            # from previous iterations, manually set them to zero at the beginning of each new training iteration.



            # Forward passing input data through the model to get predictions, start training
            
            # reshape the 3D image tensor into a 5D tensor with the shape [batch_size, channels, depth, height, width].
            # -1: The batch size dimension is inferred.
            #  1: This indicates a single channel (grayscale MRI image).
            # output = torch.utils.checkpoint.checkpoint(self.model, image.reshape(-1, 1, self.shape, self.shape, self.shape), use_reentrant=False)

            output = self.model(image.reshape(-1, 1, self.shape, self.shape, self.shape))

            labels = torch.squeeze(label)  # Squeeze the label
            # Training label shape: torch.Size([1, 256, 256, 256])
            # Squeeze labels shape : [ 256, 256, 256]

            labels = (labels * 2).round().long()  # Multiply by 2, round the values, and cast to long


            # Log the shapes and unique values of the image and label once
            if self.log_image_label_shapes_once:
                # Log image and label shapes
                self.logger.log_message(f"Training image shape: {image.shape}")
                # Training image shape: torch.Size([1, 256, 256, 256])

                self.logger.log_message(f"Training label shape: {label.shape}")
                # Training label shape: torch.Size([1, 256, 256, 256])

                self.logger.log_message(f"Training output shape: {output.shape}")
                #  Training output shape: torch.Size([1, 3, 256, 256, 256])
                
                # Log the unique values in both image and label
                unique_label = torch.unique(label)
                unique_labels = torch.unique(labels)
                self.logger.log_message(f"Unique values in training GT label: {unique_label.tolist()}")
                #  Unique values in training label: [0.0, 0.5, 1.0]    <<<<<<<<<<<<<<<<< Normalized by Pratyush
                
                self.logger.log_message(f"Unique values in training sequeezed long scaled labels: {unique_labels.tolist()}")
                #  Unique values in training label: [0, 1, 2]   

                self.log_image_label_shapes_once = False  # Set to False so this is only logged once                

            # compute the loss between the predicted output and the ground truth labels
            # The label tensor is reshaped to [batch_size, depth, height, width] to match the output shape of the model.
            # .long() * 2: Double the label values. Looks like Pratyush made it 0, 0.5, 1 range. 

            #  For CrossEntropyLoss, the input (predictions) should have 
            #  shape [batch_size, num_classes, height, width, depth] (as the output does),
            #  while the target (label) should have shape [batch_size, height, width, depth] containing class indices as integers.                



            loss = self.criterion(output, labels.reshape(-1, self.shape, self.shape, self.shape))
            train_dice = torch.mean(faster_dice(torch.squeeze(torch.argmax(output, dim=1)).long(), labels, labels=[0, 1, 2]))
            # Scale loss and backward pass (Backpropagation)
            # Backward pass (gradient calculation): Calculate the gradients for all the model's parameters with respect to the loss:
            loss.backward() #  calculate gradients
            self.optimizer.step() # gradients are applied to the model parameters 
            # Training done for that round.  

            # Accumulate loss
            total_loss += loss.item()

            # Accumulate training dice            
            total_train_dice += train_dice.item()

            if self.scheduler._step_count < self.scheduler.total_steps:
                # Update the learning rate 
                self.scheduler.step()
        
        # End of one Epoch training


        # Validation phase
        self.model.eval()
        val_loss = 0.0
        dice_scores = []
        with torch.no_grad():
            for input, label in self.validloader:
                input, label = input.to(self.device), label.to(self.device)
                output = self.model(input.reshape(-1, 1, self.shape, self.shape, self.shape))

                label = torch.squeeze(label)
                label = (label * 2).round().long()

                loss = self.criterion(output,  label.reshape(-1, self.shape, self.shape, self.shape))
                val_loss += loss.item()

                # Calculate Dice score
                pred = torch.squeeze(torch.argmax(output, dim=1))
                dice = torch.mean(faster_dice(pred, label, labels=[0, 1, 2]))
                dice_scores.append(dice)                

        # Average Dice score 
        avg_train_dice = total_train_dice / len(self.trainloader)
        avg_dice_score = sum(dice_scores) / len(dice_scores) if dice_scores else 0

        self.logger.log_message(f"{self.site_name} - Epoch [{self.current_epoch+1}/{self.total_epochs}],   Train Loss: {total_loss/len(self.trainloader):.4f}, "
              f"Val Loss: {val_loss/len(self.validloader):.4f},   Train Dice: {avg_train_dice:.4f},   Val Dice: {avg_dice_score:.4f}, lr: {self.optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint every few epochs
        # if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
        # if (self.current_epoch + 1) == self.total_epochs:
        #     save_checkpoint(self.model, self.optimizer, self.scheduler, self.current_epoch + 1)
        #     logger.log_message(f"{self.site_name} - All training  epochs finished and model saved. Stopping training.")

        # # Stop training if target Dice score is reached
        # if avg_dice_score >= self.target_dice:
        #     self.stop_training = True
        #     save_checkpoint(self.model, self.optimizer, self.scheduler, self.current_epoch + 1)  # Save the final checkpoint
        #     logger.log_message(f"{self.site_name} -Target Dice Score {self.target_dice} reached and model saved. ")             
            


        #     # # Accumulate gradients without updating yet
        #     # if (batch_id + 1) % self.gradient_accumulation_steps == 0:
        #     #     # Update optimizer
        #     #     self.scaler.step(self.optimizer)
        #     #     self.scaler.update()
        #     #     self.optimizer.zero_grad()


        # # Get the current learning rate
        # current_lr = self.optimizer.param_groups[0]['lr']

        # # Check if the learning rate has changed, and log it if so
        # if current_lr != self.previous_lr:
        #     self.logger.log_message(f"Per epoch, learning rate changed from {self.previous_lr} to: {current_lr}")
        #     self.previous_lr = current_lr  # Update the previous learning rate



        # # Log the average loss and Dice score per epoch
        # average_loss = total_loss / len(self.trainloader)

        # avg_train_dice = total_train_dice / len(self.trainloader)

        # # dice_score = self.calculate_dice(self.trainloader)
        # # Calculate Dice score on the validation set
        # # self.model.eval()  # Set the model to evaluation mode for validation
        # val_dice_score = self.calculate_dice(self.validloader, fl_ctx)  # Use validation set        
        # self.logger.log_message(f"{self.site_name} - Epoch {self.current_epoch}: Loss = {average_loss}, Avg Train Dice ={avg_train_dice}, Val Dice = {val_dice_score}")





        self.logger.log_message(f"{self.site_name} Preparing payload after epoch {self.current_epoch}")
        # return [grad.clone().cpu().numpy() for grad in gradient_accumulator if grad is not None]

        # Accumulate gradients
        gradients = []
        for i, param in enumerate(self.model.parameters()):
            if param.grad is not None:
                gradients.append(param.grad.clone().cpu().numpy())


        # Converts CUDA tensors to NumPy arrays directly isn’t allowed. We need to move tensors to the CPU before calling .numpy().
        # local_gradients = [param.grad.clone().cpu() for param in self.model.parameters()] # Move gradients to CPU
        # numpy_arrays = [tensor.numpy() for tensor in local_gradients]
        # gradients = [array.tolist() for array in numpy_arrays]
              
        

        # for example :
        # gradients = [
        #     array([[ 0.01, -0.02], [ 0.03,  0.04]]),  # gradient for some weight matrix (e.g shape [2, 2])
        #     array([0.005, -0.015]),                   # gradient for some bias vector (e.g shape [2])
        #     array([[ 0.02, 0.01], [-0.03, 0.05]])     # Another gradient for a different weight matrix and so on.
        # ]



        return gradients  



    # New one with fast dice
    def calculate_dice(self, loader, fl_ctx):
        dice_total = 0.0
        
        for image, label in loader:
            image, label = image.to(self.device), label.to(self.device)
            with torch.inference_mode():
                # Ensure consistency by reshaping image and label similarly as in the training loop
                output = self.model(image.reshape(-1, 1, self.shape, self.shape, self.shape))  # Model expects this reshaped
                # output shape : [1, 3, 256, 256, 256]

                # GT label shape: [1, 256, 256, 256]
                # Max voxel value in  GT label: 1.0

                # Squeeze and get argmax to produce predictions
                output_label = torch.squeeze(torch.argmax(output, dim=1)).long()
                # result  shape: [256, 256, 256]
                #  Max voxel value in  result: 2

                labels = torch.round(torch.squeeze(label) * 2).long()
                # labels  shape: [256, 256, 256]
                #  Max voxel value in  labels : 2

                # Compute DICE using faster_dice method and torch.mean for averaging across classes
                dice_score = torch.mean(faster_dice(output_label, labels, labels=[0, 1, 2]))

                dice_total += dice_score.item()

                # Only save one NIfTI output if the Dice score is above the threshold AND for only once
                if dice_score.item() >= self.dice_threshold_to_save_output and not self.nifti_saved:
                    self.logger.log_message("Saving NIfTI files for one input, output and label...")

                    # Get the output directory path
                    output_dir = get_output_directory_path(fl_ctx)

                    # Format the Dice score to 4 decimal places
                    dice_score_str = f"{dice_score.item():.4f}"                    
                    
                    # set paths
                    input_nifti_path = os.path.join(output_dir, f"input_sample.nii.gz") 
                    output_label_nifti_path = os.path.join(output_dir, f"output_label_dice_{dice_score_str}.nii.gz") 
                    label_nifti_path = os.path.join(output_dir, f"sq_long_by2_label_sample.nii.gz")                     


                    # Save input image (floating point), squeeze to make it 3D [256, 256, 256]
                    input_image_nifti = nib.Nifti1Image(image.squeeze(0).cpu().numpy().astype(np.float32), np.eye(4))
                    nib.save(input_image_nifti, input_nifti_path)


                    # Convert to compatible data type (int32) before saving as NIfTI
                    output_label_nifti = nib.Nifti1Image(output_label.cpu().numpy().astype(np.int32), np.eye(4))
                    label_nifti = nib.Nifti1Image(labels.cpu().numpy().astype(np.int32), np.eye(4))

                    nib.save(output_label_nifti, output_label_nifti_path)
                    nib.save(label_nifti, label_nifti_path)

                    self.logger.log_message(f"NIfTI files saved: input_sample.nii.gz, output_label_dice_{dice_score_str}.nii.gz, sq_long_by2_label_sample.nii.gz")
                    self.logger.log_message(f"Saved samples location : {output_dir}")

                    self.nifti_saved = True  # Set flag to true to avoid saving multiple times



                if self.log_shapes_once: 

                    self.logger.log_message(f"loaded image tensor shape: {image.shape}")
                    self.logger.log_message(f"Max voxel value in loaded image: {image.max().item()}") 

                    self.logger.log_message(f"Model output tensor shape: {output.shape}")
                    # output shape : [1, 3, 256, 256, 256]

                    # Log the shape after applying argmax only once
                    self.logger.log_message(f"Output label shape after argmax: {output_label.shape}")
                    # output_label  shape: [256, 256, 256]
                    
                    # Log max voxel value
                    self.logger.log_message(f"Max voxel value in  output_label: {output_label.max().item()}") 
                    #  Max voxel value in  output_label : 2 

                    # Log the GT label shape 
                    self.logger.log_message(f" GT Label shape : {label.shape}")
                    # GT label shape: [1, 256, 256, 256]

                    # Log the Squeezed long GT label shape 
                    self.logger.log_message(f" Squeezed long Labels shape : {labels.shape}")
                    # GT label shape: [ 256, 256, 256]

                    # Log the unique values in both output_label and label
                    unique_output = torch.unique(output_label)
                    unique_label = torch.unique(label)  # GT label
                    unique_labels = torch.unique(labels) # Squeezed long GT Label
                    self.logger.log_message(f"Unique values in output_label: {unique_output.tolist()}")
                    self.logger.log_message(f"Unique values in GT label: {unique_label.tolist()}")
                    self.logger.log_message(f"Unique values in Squeezed long labels: {unique_labels.tolist()}")

                    self.log_shapes_once = False 

        return dice_total / len(loader)  






    def apply_gradients(self, aggregated_gradients, fl_ctx):
        # Apply aggregated gradients to the model parameters
        # with torch.no_grad():
        #     for param, grad in zip(self.model.parameters(), aggregated_gradients):
        #         param.grad = torch.tensor(grad).to(self.device)
        #     self.optimizer.step()

        # Apply aggregated gradients to the model parameters

        aggregated_gradients = [np.array(array) for array in aggregated_gradients] 

        # self.optimizer.zero_grad()

        # The loop for param, grad in zip(self.model.parameters(), aggregated_gradients) is iterating 
        # through the model's parameters and the aggregated gradients.
        # for param, grad in zip(self.model.parameters(), aggregated_gradients):
        #     if grad is not None:
        #         param.grad = torch.tensor(grad).to(self.device) # manually setting the .grad attribute of each model parameter with the aggregated gradient.

  
        # self.optimizer.zero_grad() # ensure that any previously accumulated gradients are cleared before applying new ones

        # Apply each aggregated gradient to the corresponding model parameter
        for param, avg_grad in zip(self.model.parameters(), aggregated_gradients):
            if param.requires_grad:
                avg_grad = torch.tensor(avg_grad).to(param.device)
                avg_grad = avg_grad.to(param.grad.dtype)
                param.grad = avg_grad

        # Update model parameters based on the applied gradients
        self.optimizer.step()        

        # Clear GPU memory cache after applying gradients
        # torch.cuda.empty_cache()

        # Log the gradient application step
        self.logger.log_message(f"{self.site_name} Aggregated gradients applied to the model.")

        # Determine save points based on the total number of epochs
        if self.total_epochs < 5:
            # If fewer than 5 epochs, save at every epoch
            save_points = list(range(self.total_epochs))
        else:
            # Otherwise, save at the beginning, middle, and end, plus two other intervals
            save_points = [0, self.total_epochs // 4, self.total_epochs // 2, 3 * self.total_epochs // 4, self.total_epochs - 1]

        # Check if the current epoch is one of the save points
        if self.current_epoch in save_points:
            # Get the output directory path
            output_dir = get_output_directory_path(fl_ctx)
            
            # Save the model at the specified intervals
            model_save_path = os.path.join(output_dir, f"model_epoch_{self.current_epoch}.pth")
            torch.save(self.model.state_dict(), model_save_path)
            
            # Log the model saving step
            self.logger.log_message(f"Model saved at {model_save_path}")

        # Increment the epoch counter after processing
        self.current_epoch += 1

        # self.current_iteration += 1 

