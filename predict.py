# -*- coding: utf-8 -*-

# Imports here
import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__
from os import listdir
import argparse

import numpy as np
import torch

import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

#ESTE CODIGO DEBIERA EJECUTARSE INGRESANDO LOS 2 ARGUMENTOS CREADOS
#python predict.py --load_dir chekpoints/ --top_k 5

#-------------------------------------------------------------------------------------------------------------------------------
#PASO 0.1: RUNTIME

    #Capturar el tiempo de inicio del programa y luego el final, para restarlos y calcular el tiempo de runtime
    #La funcion sleep nos sirve para probar esto, pues duerme la ejecucion los segundos que le digamos

print("PASO 0: MEDIMOS TIEMPO INICIO")

from time import time, sleep

start_time = time()
print (start_time)

#sleep(3)

#--------------------------------------------------------------------------------------------------------------------------------

#PASO 1: CREAR UN ARRAY DE INPUT NECESARIOS Y CHEQUEARLO
print("")
print("PASO 1: RECIBIMOS LOS INPUTS DE CONFIGURACION")

def get_input_args():

    # SE CREA UN OBJETO PARA LOS ARGUMENTOS
	parser = argparse.ArgumentParser()

    ##AGREGAMOS EL ARGUMENTO 1: DIRECTORIO DEL GUARDADO DE CHECKPOINTS
	parser.add_argument('--load_dir', type = str, default = 'chekpoints/',  help = 'Directorio donde esta alojado el Checkpoint')
    ##AGREGAMOS EL ARGUMENTO 2: ARQUITECTURA DEL MODELO CNN
	parser.add_argument('--top_k', type = int, default = '5', help = 'K muestra de clasificaciones mas probables' )

    # METEMOS TODO EL PARSE_ARGS() EN UNA VARIABLE
	in_args = parser.parse_args()

	
    # VEMOS SI ESTA BIEN DEFINIDOS LOS ARGUMENTOS
	print("Argument 1:", in_args.load_dir)
	print("Argument 2:", in_args.top_k)

    #SE REEMPLAZA LA SALIDA DE LA FUNCiON QUE ESTABA EN "NONE" A LOS ARGUMENTOS QUE HICIMOS
	return parser.parse_args()
	
	
#ESTA FUNCION CHECKEA QUE HICIMOS BIEN LA FUNCION DE LOS ARGUMENTOS ANTERIORES
def check_command_line_arguments(in_arg):

    if in_arg is None:
        print("CHEQUEO - NO EXISTEN ARGUMENTOS DEFINIDOS")
    else:
        print("CHEQUEO - ARGUMENTOS DEFINIDOS")

#OCUPAMOS LAS DOS FUNCIONES ANTERIORES
in_args = get_input_args()
check_command_line_arguments(in_args)

#----------------------------------------------------------------------------------------------------

#PASO 2: CARGARMOS NUESTRO MODELO PREVIAMENTE GUARDADO PARA PODER UTILIZARLO
print("")
print("PASO 2: CARGARMOS NUESTRO MODELO PREVIAMENTE GUARDADO PARA PODER UTILIZARLO")

def load_checkpoint(filepath):
    
	checkpoint = torch.load(filepath)
	
	
	if checkpoint['arch'] == "resnet18" :
		model = models.resnet18(pretrained=True)
	else:
		model = models.vgg16(pretrained=True)
	
	model.arch = checkpoint['arch']
	model.class_to_idx = checkpoint['class_to_idx']
	model.classifier = checkpoint['classifier']
	model.load_state_dict(checkpoint['state_dict'])
	
	#optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)
	#optimizer.load_state_dict(checkpoint['optimizer_dict'])
    
	for param in model.parameters():
		param.requires_grad = False

	return model

model = load_checkpoint(in_args.load_dir+'checkpoint.pth')
print("MODELO CARGADO:" , model)
print("MODELO ARQUITECTURA:" , model.arch)


#----------------------------------------------------------------------------------------------------

#PASO 3: TRANSFORMA UNA IMAGEN CON LOS MISMAS CARACTERISTICAS DE ENTRADA DE NUESTRA RED
print("")
print("PASO 3: TRANSFORMA UNA IMAGEN CON LOS MISMAS CARACTERISTICAS DE ENTRADA DE NUESTRA RED")

from PIL import Image

img_path = "flowers/test/7/image_07215.jpg"

def process_image(image):
    
	#REALIZAMOS LA TRANSFORMACION Y NORMALIZACION DE LA IMAGEN A UN NUMPY ARRAY
    img_original = Image.open(image)
    img_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    img_procesada = img_transform(img_original)

    return img_procesada

print(process_image(img_path))

#----------------------------------------------------------------------------------------------------

#PASO 4: HACEMOS UNA FUNCION QUE RECIBE UNA IMAGEN PROCESADA Y LA MUESTRA PARA PROBAR EL PROCESAMIENTO
print("")
print("PASO 4: HACEMOS UNA FUNCION QUE RECIBE UNA IMAGEN PROCESADA Y LA MUESTRA PARA PROBAR EL PROCESAMIENTO")


def imshow(image, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    #image = image.numpy()[0].transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

#EN LA TERMINAL NO LO VAMOS A EJECUTAR PUES DA ERROR AL QUERER PLOTEAR
#imshow(process_image(img_path));
print("EN LA TERMINAL NO LO VAMOS A EJECUTAR PUES DA ERROR AL QUERER PLOTEAR")

#----------------------------------------------------------------------------------------------------
#PASO 5: USAMOS EL MODELO SOBRE NUESTRA IMAGEN PROCESADA Y CALCULAMOS LAS K CATEGORIAS MAS PROBABLES
print("")
print("PASO 5: USAMOS EL MODELO SOBRE NUESTRA IMAGEN PROCESADA Y CALCULAMOS LAS K CATEGORIAS MAS PROBABLES")


def predict(img_path, model, topk=5):  
    
    img_procesada = process_image(img_path)    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    img_procesada = img_procesada.to(device)
    
    print(type(img_procesada), img_procesada.shape)

    img_torch = img_procesada.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())

    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)

print(predict(img_path,model))


