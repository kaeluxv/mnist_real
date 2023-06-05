import torch, os, time, random, generator, discri, classify, utils, pickle
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls
from utils import *
import torch.optim as optim

class MyModel(nn.Module):
	def __init__(self, num_classes = 10):
		super(MyModel, self).__init__()
		self.feat_dim = 256
		self.num_classes = num_classes
		self.feature = nn.Sequential(
              nn.Conv2d(1, 64, 7, stride=1, padding=1),
              nn.BatchNorm2d(64),
              nn.LeakyReLU(0.2),
              nn.MaxPool2d(2, 2),
              nn.Conv2d(64, 128, 5, stride=1),
              nn.BatchNorm2d(128),
              nn.LeakyReLU(0.2),
              nn.MaxPool2d(2, 2),
              nn.Conv2d(128, 256, 5, stride=1),
              nn.BatchNorm2d(256),
              nn.LeakyReLU(0.2))
		self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
	# def forward(self, x):
	# 	x = self.feature(x)
	# 	x = x.view(x.size(0), -1)
	# 	out = self.fc_layer(x)
	# 	return out
		
	def forward(self, x):
		x = self.feature[:4](x)
	#	x = x.view(x.size(0), -1)
	#	out = self.fc_layer(x)
	#	return out
		return x

device = "cuda"
num_classes = 10
log_path = "../attack_logs"
os.makedirs(log_path, exist_ok=True)

def inversion(G, D, T, E, iden, index,output1, lr=0.2, momentum=0.9, lamda=1, iter_times=2000, clip_range=1):
#	iden = iden.cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	bs = iden.shape[0]
#	print("iden_size : " ,iden.size())
#	print("bs: ",bs)

	G.eval()
	D.eval()
	T.eval()
	E.eval()

	max_score = torch.zeros(bs)
	max_iden = torch.zeros(bs)
	z_hat = torch.zeros(bs, 100)
	final_loss = 2000
	for random_seed in range(10):
		tf = time.time()
		
		torch.manual_seed(random_seed) 
		torch.cuda.manual_seed(random_seed) 
		np.random.seed(random_seed) 
		random.seed(random_seed)

		z = torch.randn(bs, 100).cuda().float()
		z.requires_grad = True
		v = torch.zeros(bs, 100).cuda().float()
		
		for i in range(iter_times):
			fake = G(z)
			label = D(fake)
			out = T(fake)
			#out = out.float()	
			if z.grad is not None:
				z.grad.data.zero_()

			Prior_Loss = -label.mean()

			Iden_Loss = criterion(out, iden)
			Total_Loss = Prior_Loss + lamda* Iden_Loss
			Total_Loss.backward()			

			v_prev = v.clone()
			gradient = z.grad.data
			v = momentum * v - lr * gradient
			z = z + ( - momentum * v_prev + (1 + momentum) * v)
			z = torch.clamp(z.detach(), -clip_range, clip_range).float()
			z.requires_grad = True
			
			Prior_Loss_val = Prior_Loss.item()
			Iden_Loss_val = Iden_Loss.item()

			#if (i+1) % 50 == 0:
			#Total_Loss_val = Prior_Loss_val+Iden_Loss_val
			if(Iden_Loss_val<final_loss):
				final_z_num = random_seed
				final_fake = G(z.detach())
				final_iteration = i+1
				final_prior_loss =Prior_Loss_val
				final_iden_loss = Iden_Loss_val
				final_loss = Iden_Loss_val
	
	eval_prob = E(final_fake)[-1]
	eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
	final_E = eval_iden
	print("z_num:{}\tIteration:{}\tE:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}".format(final_z_num, final_iteration, final_E, final_prior_loss, final_iden_loss))
	root_path = "./Attack"
	save_img_dir = os.path.join(root_path, "GMI_imgs_mid")
	os.makedirs(save_img_dir, exist_ok=True)
	save_tensor_images(final_fake.detach(), os.path.join(save_img_dir, "attack_image{}.png".format(index)), nrow = 10)

	if(output1==final_E):
		print("correct")
		return 1
	else:
		print("not correct")
		return 0


file = "./MNIST.json"
args = load_json(json_file=file)
file_path = args['dataset']['0to9_file_path']
dataloader = init_dataloader(args, file_path, batch_size=1, mode ="gan")
		
my_model = MyModel(10).to(device)
my_model.load_state_dict(torch.load('./mcnn_dict_state.tar')['state_dict'])
my_model.eval()


if __name__ == "__main__":
	target_path = "./mcnn_dict_state.tar"

	T = classify.MCNN(num_classes)
	T = nn.DataParallel(T).cuda()
	ckp_T = torch.load(target_path)['state_dict']
	new_state_dict={}
	for key, value in ckp_T.items():
		new_key = 'module.' + key
		new_state_dict[new_key] = value
	utils.load_my_state_dict(T, new_state_dict)
#	T.load_state_dict(torch.load(target_path)['state_dict'])
#	print("1")

	e_path = "./scnn_dict_state.tar"
	E = classify.SCNN(num_classes)
	E = nn.DataParallel(E).cuda()
	ckp_E = torch.load(e_path)['state_dict']
	new_state_dict={}
	for key,value in ckp_E.items():
		new_key = 'module.' + key
		new_state_dict[new_key] = value
	utils.load_my_state_dict(E, new_state_dict)

	
	g_path = "./dasol/MNIST_G.tar"
	G = generator.GeneratorMNIST()
	G = nn.DataParallel(G).cuda()
	ckp_G = torch.load(g_path)['state_dict']
	utils.load_my_state_dict(G, ckp_G)


	d_path = "./dasol/MNIST_D.tar"
	D = discri.DGWGAN32()
	D = nn.DataParallel(D).cuda()
	ckp_D = torch.load(d_path)['state_dict']
	utils.load_my_state_dict(D, ckp_D)

	correct = 0
	index=0
	for i, imgs in enumerate(dataloader):
		if(i==100):
			break
		
		imgs = imgs.to(device)
		torch.set_printoptions(threshold=np.inf)

		output = my_model.forward(imgs)
		#output = output.data.max(1)[1]
		output1 = my_model.feature[4:](output)
		output1 = output1.view(output1.size(0),-1)
		output1 = my_model.fc_layer(output1)
		output1 = output1.data.max(1)[1]
		#output1 = output
		print("i:",i,end='')
		print(" output : ", output1)
	
		torch.save(output,'./out/output{}.pt'.format(i),pickle_module = pickle)

		root_path = "./Attack"
		save_img_dir = os.path.join(root_path, "GMI_imgs_mid")
		os.makedirs(save_img_dir, exist_ok=True)
		save_tensor_images(imgs.detach(), os.path.join(save_img_dir, "origin_image{}.png".format(i)), nrow = 10)

		iden = torch.load('./out/output{}.pt'.format(i))
		#iden = iden.float()
		iden = iden.to(device)
		
		correct += inversion(G, D, T, E, iden,i,output1)
		index+=1
		
	acc = correct/(index)
	print("total acc: ",acc)
