import torch.nn as nn
import torch.nn.functional as F


class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x) 

		return x


class ImgFGDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64, num_modes = 10):
		super(ImgFGDiscriminator, self).__init__()

		self.D = nn.Sequential(
			nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
			nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
			nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
			nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(negative_slope=0.2, inplace=True)
		)

		self.cls1 = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.cls2 = nn.Sequential(
			nn.AdaptiveAvgPool2d((10, 10)),
			nn.Linear((ndf*8) * 10 * 10, num_modes)
		)

		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		out = self.D(x)
		out_domain = self.cls1(out)
		out_mode = self.cls2(out)
		#x = self.up_sample(x)
		#x = self.sigmoid(x)

		return out_domain, out_mode
