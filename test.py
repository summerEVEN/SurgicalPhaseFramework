import torch


def cuda_test():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("GPU可用")
    else:
        device = torch.device("cpu")
        print("GPU不可用，使用CPU")

    print(device)

if __name__ == "__main__":
    cuda_test()