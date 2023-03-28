from pyexpat import model
import torch
import numpy as np
import argparse
import os
import pickle
from utils import print_and_log, get_log_files, TestAccuracies, loss, aggregate_accuracy, verify_checkpoint_dir, task_confusion
from model import SloshNet
import torch.nn.functional as F  
import Utils.optimizer as optim
import torchvision
import video_reader
import random 
from tqdm import tqdm

def main():
    learner = Learner()
    learner.run()

class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, False)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)
        
        gpu_device = 'cuda'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()

        self.train_set, self.validation_set, self.test_set = self.init_data()

        self.vd_train = video_reader.VideoDataset(self.args, train_mode= True)
        self.vd_test = video_reader.VideoDataset(self.args, train_mode=False)
        self.video_loader_train = torch.utils.data.DataLoader(self.vd_train, batch_size=1, num_workers=self.args.num_workers,pin_memory=True )
        self.video_loader_train_dt = torch.utils.data.DataLoader(self.vd_train, batch_size=1, num_workers=self.args.num_workers,pin_memory=True )
        self.video_loader_test = torch.utils.data.DataLoader(self.vd_test, batch_size=1, num_workers=self.args.num_workers, pin_memory=True)
        
        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        num_param = list(self.model.darts_model.alphas()) + list(self.model.beta1) +list(self.model.gamma1)
        alpha_param =  list(map(id, num_param)) 
        select_params =filter(lambda p: id(p)  in alpha_param,
                                  self.model.parameters())
        model_params = filter(lambda p: id(p) not in alpha_param,
                                  self.model.parameters())

        if self.args.opt == "adam":
            self.optimizer = torch.optim.Adam(model_params, 3e-3, betas=(0.5, 0.999),
                                   weight_decay=1e-3)
            self.alpha_optim = torch.optim.Adam(self.model.darts_model.alphas(), 3e-3, betas=(0.5, 0.999),
                                   weight_decay=1e-3)

        elif self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(model_params, lr=self.args.learning_rate,momentum=0.9, weight_decay=3e-4 )

            self.alpha_optim = torch.optim.Adam(select_params,3e-3, betas=(0.5, 0.999),
                                   weight_decay=1e-3)
        self.test_accuracies = TestAccuracies(self.test_set)       
        
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()
        self.alpha_optim.zero_grad()

    def init_model(self):
        model = SloshNet(self.args)
        model = model.to(self.device) 
        if self.args.num_gpus > 1:
            model.distribute_model()
        return model

    def init_data(self):
        train_set = [self.args.dataset]
        validation_set = [self.args.dataset]
        test_set = [self.args.dataset]
        return train_set, validation_set, test_set   

    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", choices=["ssv2", "kinetics", "hmdb", "ucf"], default="ssv2", help="Dataset to use.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16, help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default=None, help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--training_iterations", "-i", type=int, default=100020, help="Number of meta-training iterations.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False, action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of each task.")
        parser.add_argument("--shot", type=int, default=5, help="Shots per class.")
        parser.add_argument("--query_per_class", type=int, default=5, help="Target samples (i.e. queries) per class used for training.")
        parser.add_argument("--query_per_class_test", type=int, default=1, help="Target samples (i.e. queries) per class used for testing.")
        parser.add_argument('--test_iters', nargs='+', type=int, help='iterations to test at. Default is for ssv2 otam split.', default=[75000])
        parser.add_argument("--num_test_tasks", type=int, default=10000, help="number of random tasks to test on.")
        parser.add_argument("--print_freq", type=int, default=1000, help="print and log every n iterations.")
        parser.add_argument("--seq_len", type=int, default=8, help="Frames per video.")
        parser.add_argument("--num_workers", type=int, default=10, help="Num dataloader workers.")
        parser.add_argument("--method", choices=["resnet18", "resnet34", "resnet50", "resnet50_darts"], default="resnet50_darts", help="method")
        parser.add_argument("--trans_linear_out_dim", type=int, default=1152, help="Transformer linear_out_dim")
        parser.add_argument("--opt", choices=["adam", "sgd", "adamw", "new"], default="sgd", help="Optimizer")
        parser.add_argument("--trans_dropout", type=int, default=0.1, help="Transformer dropout")
        parser.add_argument("--save_freq", type=int, default=5000, help="Number of iterations between checkpoint saves.")
        parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")
        parser.add_argument('--temp_set', nargs='+', type=int, help='cardinalities e.g. 2,3 is pairs and triples', default=[2])
        parser.add_argument("--scratch", choices=["bc", "bp"], default="bp", help="directory containing dataset, splits, and checkpoint saves.")
        parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to split the ResNet over")
        parser.add_argument("--debug_loader", default=False, action="store_true", help="Load 1 vid per class for debugging")
        parser.add_argument("--split", type=int, default=7, help="Dataset split.")
        parser.add_argument('--sch', nargs='+', type=int, help='iters to drop learning rate', default=[1000000])
        parser.add_argument("--test_model_only", type=bool, default=False,help="Only testing the model from the given checkpoint")
        parser.add_argument("--weight_decay", type=float, default=5e-5, help="weight decay")
        parser.add_argument("--step_iterations",'-step_iter',type=int, default=1000, help="step iterations")
        parser.add_argument("--steps",type=int, nargs='+', default=[0, 6, 9], help="LRS")
        parser.add_argument("--LRS",type=float, nargs='+', default=[1, 0.1, 0.01], help="steps")
        parser.add_argument("--warmup_epochs", type=int, default=1, help="warmup_epochs")
        parser.add_argument("--warmup_start_lr", type=float, default=0.0001, help="warmup_start_lr")

        args = parser.parse_args()
        

        args.scratch =  './' 
        if args.checkpoint_dir == None:
            print("need to specify a checkpoint dir")
            exit(1)

        if (args.method == "resnet50") or (args.method == "resnet34"):
            args.img_size = 224
        if args.method == "resnet50" or args.method == "resnet50_darts":
            args.trans_linear_in_dim = 2048
        else:
            args.trans_linear_in_dim = 512
        
        if args.dataset == "ssv2":
            args.traintestlist = os.path.join(args.scratch, "./splits/ssv2_OTAM")
            args.path = '/public/datasets/few_shot/smsm_otam_extracted_frames'
        elif args.dataset == "kinetics":
            args.traintestlist = os.path.join(args.scratch, "./splits/kinetics_CMN")
            args.path = '/public/datasets/few_shot/kinetics_100_extracted_frames'
        elif args.dataset == "ucf":
            args.traintestlist = os.path.join(args.scratch, "./splits/ucf_ARN")
            args.path = '/public/datasets_neo/ucf101/extracted_frames'
        elif args.dataset == "hmdb":
            args.traintestlist = os.path.join(args.scratch, "./splits/hmdb_ARN")
            args.path ='/public/datasets_neo/hmdb51/extracted_frames'
        return args

    def run(self):
        train_accuracies = []
        losses = []
        total_iterations = self.args.training_iterations
        iteration = self.start_iteration

        step_iterations = self.args.step_iterations
        cur_epoch = iteration// step_iterations

        if self.args.test_model_only:  
            print("Model being tested at path: " + self.args.test_model_path)
            self.load_checkpoint()
            accuracy_dict = self.test()
            print(accuracy_dict)

        for (task_dict,task_dict_dt) in tqdm(zip(self.video_loader_train,self.video_loader_test)):

            if iteration >= total_iterations:
                break
            iteration += 1
            torch.set_grad_enabled(True)

            task_loss, task_accuracy = self.train_task(task_dict)
            train_accuracies.append(task_accuracy)
            losses.append(task_loss)
            
            for param_idx, param_group in enumerate(self.optimizer.param_groups):
                old_lr =  self.args.learning_rate
                lr = optim.get_epoch_lr(cur_epoch +  float(iteration) / step_iterations,  self.args, old_lr)
                optim.set_lr(param_group, lr)

            for param_idx, param_group in enumerate(self.alpha_optim.param_groups):
                alpha_old_lr =  3e-3
                alpha_lr = optim.get_epoch_lr(cur_epoch +  float(iteration) / step_iterations,  self.args, alpha_old_lr)
                optim.set_lr(param_group, alpha_lr)

            # optimize   
            if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.alpha_optim.step()
                self.alpha_optim.zero_grad()
           
            if (iteration + 1) % self.args.print_freq == 0:
                # print training stats
                print_and_log(self.logfile,'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                                .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                        torch.Tensor(train_accuracies).mean().item()))
                train_accuracies = []
                losses = []
                for p in self.model.darts_model.alphas():
                    print_and_log(self.logfile,'darts_model.alphas:{}'.format(F.softmax(p, dim=-1)))
                print_and_log(self.logfile,
                              'darts_model.beta1:{}'.format(F.softmax(self.model.beta1.data, dim=-1)))
                print_and_log(self.logfile,
                              'darts_model.gamma1:{}'.format(F.softmax(self.model.gamma1.data, dim=-1)))
                optimizers=[self.optimizer, self.alpha_optim]
                for param_id,optimizer in enumerate(optimizers):
                    for param_group in optimizer.param_groups:
                        print_and_log(self.logfile,str(param_id)+":{}".format(param_group['lr']))

            if ((iteration + 1) % self.args.save_freq == 0) and (iteration + 1) != total_iterations:
                self.save_checkpoint(iteration + 1)

            if ((iteration + 1) % self.args.test_iters[0] == 0) and (iteration + 1) != total_iterations:
                accuracy_dict = self.test()
                print(accuracy_dict)
                self.test_accuracies.print(self.logfile, accuracy_dict)

        # save the final model
        torch.save(self.model.state_dict(), self.checkpoint_path_final)

        self.logfile.close()

    def train_task(self, task_dict):
        self.model.train()
        context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list, real_support_labels  = self.prepare_task(task_dict)

        model_dict = self.model(context_images, context_labels, target_images)
        target_logits = model_dict['logits'].to(self.device)

        task_loss = self.loss(target_logits, target_labels, self.device)  / self.args.tasks_per_batch
        task_accuracy = self.accuracy_fn(target_logits, target_labels)

        task_loss.backward(retain_graph=False)

        return task_loss, task_accuracy

    def test(self):
        self.model.eval()
        with torch.no_grad():
            accuracy_dict ={}
            accuracies = []
            iteration = 0
            item = self.args.dataset
            for task_dict in tqdm(self.video_loader_test):
                if iteration >= self.args.num_test_tasks:
                    break
                iteration += 1

                context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list, real_support_labels  = self.prepare_task(task_dict)
                model_dict = self.model(context_images, context_labels, target_images)
                target_logits = model_dict['logits'].to(self.device)
                accuracy = self.accuracy_fn(target_logits, target_labels)
                accuracies.append(accuracy.item())
                del target_logits

            accuracy = np.array(accuracies).mean() * 100.0
            confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

            accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}
            

        return accuracy_dict


    def prepare_task(self, task_dict, images_to_device = True):
        context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
        target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]
        real_target_labels = task_dict['real_target_labels'][0]
        batch_class_list = task_dict['batch_class_list'][0]
        real_support_labels = task_dict['real_support_labels'][0]

        if images_to_device:
            context_images = context_images.to(self.device)
            target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list, real_support_labels  

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]


    def save_checkpoint(self, iteration):
        d = {'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()}

        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint{}.pt'.format(iteration)))
        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        if self.args.test_model_only:
            checkpoint = torch.load(self.args.test_model_path)
        else:
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint.pt'))     
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    main()