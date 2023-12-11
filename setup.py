import torch

cudaAvailability = torch.cuda.is_available()
print("Cuda availability: {0}".format(cudaAvailability) )
print("Current pytorch version is: " + torch.__version__)
print("Current version of Cuda is: " + torch.version.cuda)

def useGPU():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using", device, "device")

def useCPU():
    device = torch.device("cpu")
    print("Using", device, "device")

useGPU()
#x = torch.rand(5, 3)
#print(x)

print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
