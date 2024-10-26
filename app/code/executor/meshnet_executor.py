import torch
import os
import random  
import numpy as np  
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
            self.model.load_state_dict(torch.load(initial_weights_file_path))
            self.logger.log_message(f"Loaded initial weights from file for this site.")
        else:
            # Save initial weights to file (first site)
            torch.save(self.model.state_dict(), initial_weights_file_path)
            self.logger.log_message(f"Initial weights saved to file for this site.")        


        # Ensure model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True

        # Optimizer and criterion setup
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003) #<<<<<<<<
        self.criterion = torch.nn.CrossEntropyLoss()

        # Add learning rate scheduler ( Overlook for now)     <<<<<<<<<<
        # This will reduce the learning rate by a factor of 0.1 if the validation loss does not improve for 5 epochs.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)




        self.current_epoch = 0

        # Epochs and aggregation interval
        self.total_epochs = 30  # Set the total number of epochs          #<<<<<<<<
        # self.aggregation_interval = 1  # Aggregation occurs every 5 epochs (you can modify this)

        self.dice_threshold = 0.9  # Set the Dice score threshold

        # Gradient accumulation  ( Overlook for now)  <<<<<<<<<<
        # self.gradient_accumulation_steps = 4

        # Early stopping   ( Overlook for now)  <<<<<<<<<<
        # self.early_stopping_patience = 10
        # self.best_loss = float('inf')
        # self.epochs_without_improvement = 0

        # Initialization flag for data loader
        self.data_loader_initialized = False

    def set_seed(self, seed):
        # Set the random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
        torch.backends.cudnn.benchmark = False

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

            # Initialize Data Loader with dynamic path
            self.data_loader = Scanloader(db_file=db_file_path, label_type='GWlabels', num_cubes=1)
            self.trainloader, self.validloader, self.testloader = self.data_loader.get_loaders(batch_size=1)
            

            # Set flag to True so the loader is only initialized once
            self.data_loader_initialized = True
            self.logger.log_message(f"Data loader initialized for site {self.site_name}")


        if task_name == "train_and_get_gradients":
            self.logger.log_message(f"{self.site_name}-train_and_get_gradients called ")
            gradients = self.train_and_get_gradients()
            outgoing_shareable = Shareable()
            outgoing_shareable["gradients"] = gradients
            return outgoing_shareable

        elif task_name == "accept_aggregated_gradients":
            self.logger.log_message(f"{self.site_name}-accept_aggregated_gradients called ")
            aggregated_gradients = shareable["aggregated_gradients"]
            self.apply_gradients(aggregated_gradients, fl_ctx)
            return Shareable()

    # def train_and_get_gradients_old(self):
    #     for epoch in range(self.total_epochs):

    #         self.logger.log_message(f"Starting Epoch {epoch}/{self.total_epochs}, Aggregation Interval: {self.aggregation_interval}")
    #         self.model.train()
            
    #         # Initialize accumulators for the loss and gradients
    #         total_loss = 0.0
    #         gradient_accumulator = [torch.zeros_like(param).to(self.device) for param in self.model.parameters()]
            
    #         # Training loop for one epoch (full pass through the dataset)
    #         for batch_id, (image, label) in enumerate(self.trainloader):
    #             image, label = image.to(self.device), label.to(self.device)
    #             self.optimizer.zero_grad()

    #             # Mixed precision and checkpointing
    #             with torch.amp.autocast(device_type='cuda'):
    #                 output = torch.utils.checkpoint.checkpoint(self.model, image, use_reentrant=False)
    #                 label = label.squeeze(1)
    #                 loss = self.criterion(output, label.long())

    #             total_loss += loss.item()

    #             # Scale loss and backward pass
    #             self.scaler.scale(loss).backward()

    #             # Accumulate gradients
    #             for i, param in enumerate(self.model.parameters()):
    #                 if param.grad is not None:
    #                     gradient_accumulator[i] += param.grad.clone()

    #             self.scaler.step(self.optimizer)
    #             self.scaler.update()

    #             torch.cuda.empty_cache()

    #         # Log the average loss per epoch
    #         average_loss = total_loss / len(self.trainloader)
    #         dice_score = self.calculate_dice(self.trainloader)
    #         self.logger.log_message(f"Site {self.site_name} - Epoch {epoch}: Loss = {average_loss}, Dice = {dice_score}")

    #         # Call aggregation based on your set aggregation_interval
    #         if (epoch + 1) % self.aggregation_interval == 0:
    #             # Perform model aggregation here
    #             return [grad.clone().cpu().numpy() for grad in gradient_accumulator if grad is not None]

    #     return []

    # def train_and_get_gradients_new(self):
    #     for epoch in range(self.total_epochs):
    #         # self.logger.log_message(f"Starting Epoch {epoch+1}/{self.total_epochs}, Aggregation Interval: {self.aggregation_interval}")
    #         self.model.train()

    #         # Initialize accumulators for the loss and gradients
    #         total_loss = 0.0
    #         gradient_accumulator = [torch.zeros_like(param).to(self.device) for param in self.model.parameters()]

    #         # Training loop for one epoch (full pass through the dataset)
    #         for batch_id, (image, label) in enumerate(self.trainloader):
    #             image, label = image.to(self.device), label.to(self.device)
    #             self.optimizer.zero_grad()

    #             # Mixed precision and checkpointing
    #             with torch.amp.autocast(device_type='cuda'):
    #                 output = torch.utils.checkpoint.checkpoint(self.model, image, use_reentrant=False)
    #                 label = label.squeeze(1)
    #                 loss = self.criterion(output, label.long())

    #             total_loss += loss.item()

    #             # Scale loss and backward pass
    #             self.scaler.scale(loss).backward()

    #             # Accumulate gradients
    #             for i, param in enumerate(self.model.parameters()):
    #                 if param.grad is not None:
    #                     gradient_accumulator[i] += param.grad.clone()

    #             self.scaler.step(self.optimizer)
    #             self.scaler.update()

    #             torch.cuda.empty_cache()

    #         # Log the average loss per epoch
    #         average_loss = total_loss / len(self.trainloader)
    #         dice_score = self.calculate_dice(self.trainloader)
    #         self.logger.log_message(f"Site {self.site_name} - Epoch {epoch+1}: Loss = {average_loss}, Dice = {dice_score}")

    #         # Check if it's time to perform aggregation
    #         if (epoch + 1) % self.aggregation_interval == 0:
    #             # Return the gradients after completing the specified aggregation interval
    #             self.logger.log_message(f"Performing aggregation after epoch {epoch+1}")
    #             return [grad.clone().cpu().numpy() for grad in gradient_accumulator if grad is not None]

    #     return []


    # def setup_site_logger(self):
    #     site_id = os.getenv('FL_SITE_ID', 'site_unknown')  # Use environment variable or other means to set site ID
    #     log_dir = f'logs/{site_id}'
    #     os.makedirs(log_dir, exist_ok=True)
    #     log_filename = os.path.join(log_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    #     logging.basicConfig(
    #         filename=log_filename,
    #         level=logging.INFO,
    #         format='%(asctime)s - %(levelname)s - %(message)s',
    #         datefmt='%Y-%m-%d %H:%M:%S',
    #     )

    #     self.logger = logging.getLogger()
    #     self.logger.info("Logging started")



    def load_model_weights(self, model_weights):
        # Load the received model weights into the local model
        # (((( FOR FUTUR USE))))
        self.model.load_state_dict(model_weights)
        self.logger.log_message(f"Loaded initial model weights for site {self.site_name}")



    def train_and_get_gradients(self):
        self.model.train()

        # Initialize accumulators for the loss and gradients
        total_loss = 0.0
        # gradient_accumulator = [torch.zeros_like(param).to(self.device) for param in self.model.parameters()]

        # Training loop for one epoch (full pass through the dataset)
        for batch_id, (image, label) in enumerate(self.trainloader):
            image, label = image.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()

            # Mixed precision and checkpointing
            with torch.amp.autocast(device_type='cuda'):
                output = torch.utils.checkpoint.checkpoint(self.model, image, use_reentrant=False)
                label = label.squeeze(1)
                loss = self.criterion(output, label.long())

            # Accumulate loss
            total_loss += loss.item()

            # Scale loss and backward pass
            # self.scaler.scale(loss).backward()

            # Scale loss and backward pass
            loss.backward() #  calculate gradients
            self.optimizer.step() # gradients are applied to the model parameters           

            # # Accumulate gradients without updating yet
            # if (batch_id + 1) % self.gradient_accumulation_steps == 0:
            #     # Update optimizer
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
            #     self.optimizer.zero_grad()


        # Clear GPU cache (No need)
        # torch.cuda.empty_cache()

        # Log the average loss and Dice score per epoch
        average_loss = total_loss / len(self.trainloader)
        # dice_score = self.calculate_dice(self.trainloader)
        # Calculate Dice score on the validation set
        self.model.eval()  # Set the model to evaluation mode for validation
        dice_score = self.calculate_dice(self.validloader)  # Use validation set        
        self.logger.log_message(f"{self.site_name} - Epoch {self.current_epoch}: Loss = {average_loss}, Dice = {dice_score}")

        # Update the learning rate based on the loss     <<<<<<<<<<<<<< (Overlooked for now)
        self.scheduler.step(average_loss)

        # Check for early stopping           <<<<<<<<<<<<<<<<<<  (Overlooked for now)
        # if average_loss < self.best_loss:
        #     self.best_loss = average_loss
        #     self.epochs_without_improvement = 0
        # else:
        #     self.epochs_without_improvement += 1

        # if self.epochs_without_improvement >= self.early_stopping_patience:
        #     self.logger.log_message(f"Early stopping triggered at epoch {self.current_epoch}")
        #     return []

        self.logger.log_message(f"{self.site_name} Preparing payload after an iteration in epoch {self.current_epoch}")
        # return [grad.clone().cpu().numpy() for grad in gradient_accumulator if grad is not None]

        # Accumulate gradients
        gradients = []
        for i, param in enumerate(self.model.parameters()):
            if param.grad is not None:
                gradients.append(param.grad.clone().cpu().numpy())
        

        return gradients        

    # def calculate_dice(self, loader):
    #     dice_total = 0.0
    #     for image, label in loader:
    #         image, label = image.to(self.device), label.to(self.device)
    #         with torch.no_grad():
    #             output = self.model(image)
    #             output_label = torch.argmax(output, dim=1)
    #             dice_score = faster_dice(output_label, label.squeeze(1), labels=[0, 1, 2])
    #             dice_total += dice_score.mean().item()
    #     return dice_total / len(loader)


    def calculate_dice(self, loader):
        dice_total = 0.0
        for image, label in loader:
            image, label = image.to(self.device), label.to(self.device)
            with torch.no_grad():
                output = self.model(image)
                output_label = torch.argmax(output, dim=1)
                dice_score = self.calculate_dice_score(output_label, label.squeeze(1))
                dice_total += dice_score
        return dice_total / len(loader)

    def calculate_dice_score(self, pred, target, num_classes=3):
        dice_scores = []

        for class_idx in range(num_classes):
            pred_class = (pred == class_idx).float()
            target_class = (target == class_idx).float()

            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()

            # To avoid division by zero, add a small epsilon to the denominator
            dice_score = (2.0 * intersection) / (union + 1e-6)
            dice_scores.append(dice_score.item())

        # Return the mean Dice score across all classes
        return sum(dice_scores) / len(dice_scores)



    def apply_gradients(self, aggregated_gradients, fl_ctx):
        # Apply aggregated gradients to the model parameters
        # with torch.no_grad():
        #     for param, grad in zip(self.model.parameters(), aggregated_gradients):
        #         param.grad = torch.tensor(grad).to(self.device)
        #     self.optimizer.step()

        # Apply aggregated gradients to the model parameters
        self.optimizer.zero_grad()
        for param, grad in zip(self.model.parameters(), aggregated_gradients):
            param.grad = torch.tensor(grad).to(self.device)
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

