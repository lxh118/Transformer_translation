import torch

from torch.optim import Adam
import importlib

from src.cn2en_dataset import Cn2EnDataset, cn2en_collate_fn
from transformer_class import Seq2SeqTransformer, device
from src.en2cn_dataset import En2CnDataset, get_dataloader, en2cn_collate_fn


def en2cn_run():
    """
    实现英文翻译为中文
    :return:
    """

    # 根据数据集选择要加载的配置文件
    selected_lib = "en2cn"
    lib = importlib.import_module(f"src.{selected_lib}_lib")

    lib.init_seeds(10)

    input_dim = lib.INPUT_DIM
    output_dim = lib.OUTPUT_DIM
    d_model = lib.D_MODEL
    num_heads = lib.NUM_HEADS
    ff_hidden_dim = lib.FF_HIDDEN_DIM
    num_layers = lib.NUM_LAYERS
    dropout = lib.DROPOUT

    # Create an instance of Seq2SeqTransformer
    model = Seq2SeqTransformer(input_dim, output_dim, d_model, num_heads, ff_hidden_dim, num_layers, dropout,
                               selected_lib=selected_lib).to(device)

    optimizer = Adam(model.parameters(), 0.001)

    en2cn_dataset = En2CnDataset()
    en2cn_total_size = len(en2cn_dataset)
    en2cn_train_size = int(lib.train_ratio * en2cn_total_size)
    en2cn_val_size = int(lib.val_ratio * en2cn_total_size)
    en2cn_test_size = en2cn_total_size - en2cn_train_size - en2cn_val_size

    en2cn_train_dataloader, en2cn_val_dataloader, en2cn_test_dataloader = get_dataloader(en2cn_dataset,
                                                                                         en2cn_train_size,
                                                                                         en2cn_val_size,
                                                                                         en2cn_test_size,
                                                                                         custom_collate_fn=en2cn_collate_fn)
    checkpoint = torch.load(lib.model_save_path)
    # print(checkpoint.keys())
    model.load_state_dict(checkpoint)

    model.trainer(model, lib.epochs, en2cn_train_dataloader, en2cn_val_dataloader, optimizer, lib)

    en2cn_output = model.tester(model, en2cn_test_dataloader, lib)

    return en2cn_output


def cn2en_run():
    """
    实现中文翻译为英文
    :return:
    """
    # 根据数据集选择要加载的配置文件
    selected_lib = "cn2en"  # 或者 "en2cn"
    lib = importlib.import_module(f"src.{selected_lib}_lib")

    lib.init_seeds(10)

    input_dim = lib.INPUT_DIM
    output_dim = lib.OUTPUT_DIM

    d_model = lib.D_MODEL
    num_heads = lib.NUM_HEADS
    ff_hidden_dim = lib.FF_HIDDEN_DIM
    num_layers = lib.NUM_LAYERS
    dropout = lib.DROPOUT

    # Create an instance of Seq2SeqTransformer
    model = Seq2SeqTransformer(input_dim, output_dim, d_model, num_heads, ff_hidden_dim, num_layers, dropout,
                               selected_lib=selected_lib).to(device)
    optimizer = Adam(model.parameters(), 0.001)

    cn2en_dataset = Cn2EnDataset()
    cn2en_train_size = lib.train_size
    cn2en_val_size = lib.val_size
    cn2en_test_size = lib.test_size

    cn2en_train_dataloader, cn2en_val_dataloader, cn2en_test_dataloader = get_dataloader(cn2en_dataset,
                                                                                         cn2en_train_size,
                                                                                         cn2en_val_size,
                                                                                         cn2en_test_size,
                                                                                         batch_size=lib.batch_size,
                                                                                         num_workers=lib.num_works,
                                                                                         custom_collate_fn=cn2en_collate_fn)

    model.trainer(model, lib.epochs, cn2en_train_dataloader, cn2en_val_dataloader, optimizer, lib)
    cn2en_output = model.tester(model, cn2en_test_dataloader, lib)
    return cn2en_output


# watch -n 10 nvidia-smi
if __name__ == '__main__':
    output = en2cn_run()
    print(f"test_bleu_score {output:7.4f}")

    # output = cn2en_run()
    # print(f"test_bleu_score {output:7.4f}")

    pass
