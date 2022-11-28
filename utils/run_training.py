from tensorflow.keras import optimizers, losses, metrics, Model, callbacks
from tensorflow.keras.layers import Input

from utils.logger import write_to_file
from data.make_dataset import build_dataset
from utils.train_multitask import train_model_multitask
from utils.train_singletask import train_model_singletask
from model.build_model import LeviHassnerBackbone, MultiTaskHead

def run_model_training(cfg, data_dir):
    #init dataset
    train_ds, val_ds = build_dataset(data_dir, batch_size=cfg.DATA.BATCH_SIZE)

    #init model
    print("Initializing TF Model")
    model = None
    if cfg.DATA.TARGET_LABEL in ["age", "gender"]:
        model = LeviHassnerBackbone(
            weight_decay = cfg.MODEL.OPTIMIZER.GAMMA,
            dropout_prob = cfg.MODEL.DROPOUT,
            include_head = True,
            num_classes = 1 if cfg.DATA.TARGET_LABEL == "gender" else 8,
            initializer = cfg.MODEL.INITIALIZER,
        )
    elif cfg.DATA.TARGET_LABEL == "both":
        backbone = LeviHassnerBackbone(
            weight_decay = cfg.MODEL.OPTIMIZER.GAMMA,
            dropout_prob = cfg.MODEL.DROPOUT,
            include_head = False,
            initializer = cfg.MODEL.INITIALIZER,
        )

        if cfg.MODEL.PRETRAIN_PATH is not None:
            print("Loading pretrained model from path", cfg.MODEL.PRETRAIN_PATH)
            backbone.load_weights(cfg.MODEL.PRETRAIN_PATH)
            # backbone.trainable = False
            for layer in backbone.layers:
                if "Conv" in layer.name:
                    layer.trainable=False

            print("Successfully loaded pretrained model weights for backbone.")

        head = MultiTaskHead(
            num_classes_1=8,
            num_classes_2=1,
            activation_1="softmax",
            activation_2="sigmoid",
            name_1="age",
            name_2="gender",
        )

        # input = Input(shape=cfg.MODEL.INPUT_SHAPE)
        input = Input(shape=(227, 227, 3))
        latent = backbone(input)
        output = head(latent)
        
        model = Model(inputs=input, outputs=output)
    
    model.build(input_shape=(None, 227, 227, 3))

    #Tensorboard
    tb_callback = callbacks.TensorBoard(cfg.LOG_DIR, write_images=True)
    tb_callback.set_model(model)

    with open(cfg.LOG_FILE, "a") as af:
        af.write("==========MODEL ARCHITECTURE==========\n")
        model.summary(print_fn=lambda x: af.write(x + '\n'))
        af.write("\n\n")
        print("==========MODEL ARCHITECTURE==========\n")
        model.summary(print_fn=lambda x: print(x+"\n"))

    #init optimizer or pass to train functin
    print("Initializing optimizer \n")
    optimizer = None
    if cfg.MODEL.OPTIMIZER.NAME == "adam":
        print("Using Adam")
        optimizer = optimizers.Adam(
            learning_rate = cfg.MODEL.OPTIMIZER.ALPHA,
            beta_1 = cfg.MODEL.OPTIMIZER.BETAS[0],
            beta_2 = cfg.MODEL.OPTIMIZER.BETAS[1],
            name = cfg.MODEL.OPTIMIZER.NAME,
        )
    elif cfg.MODEL.OPTIMIZER.NAME == "sgd":
        print("Using SGD")
        optimizer = optimizers.SGD(
            learning_rate = cfg.MODEL.OPTIMIZER.ALPHA,
            momentum = cfg.MODEL.OPTIMIZER.MOMENTUM,
            name = cfg.MODEL.OPTIMIZER.NAME,
        )
    
    #init loss and metrics
    criterions = []
    eval_metrics = []
    if cfg.DATA.TARGET_LABEL in ["age", "gender"]:
        criterions = [
            losses.SparseCategoricalCrossentropy() if cfg.DATA.TARGET_LABEL == "age" else losses.BinaryCrossentropy(),
        ]

        eval_metrics = [
            metrics.SparseCategoricalAccuracy() if cfg.DATA.TARGET_LABEL == "age" else metrics.BinaryAccuracy(),
        ]
    else:
        criterions = [
            losses.SparseCategoricalCrossentropy(),
            losses.BinaryCrossentropy()
        ]

        eval_metrics = [
            metrics.SparseCategoricalAccuracy(),
            metrics.BinaryAccuracy(),
        ]

    #Run training
    print("Starting training...\n\n")
    if cfg.DATA.TARGET_LABEL in ["age", "gender"]:
        with open(cfg.LOG_FILE, "a") as af:
            af.write("Training model on task %s recognition.\n" % (cfg.DATA.TARGET_LABEL))

        model, best_acc = train_model_singletask(
            model, train_ds, val_ds, 
            epochs=cfg.TRAIN.MAX_EPOCHS,
            loss_fn=criterions[0],
            acc_metric=eval_metrics[0],
            log_file=cfg.LOG_FILE,
            target_index=0 if cfg.DATA.TARGET_LABEL == "age" else 1,
            tensorboard_dir=cfg.LOG_DIR,
            tensorboard_prefix=cfg.PREFIX,
        )

        return best_acc

        # return test_model_singletask(
        #     model, val_ds,
        #     loss_fn=criterions[0],
        #     acc_metric=eval_metrics[0],
        #     target_index=0 if cfg.DATA.TARGET_LABEL == "age" else 1,
        # )[1]
    elif cfg.DATA.TARGET_LABEL == "both":
        with open(cfg.LOG_FILE, "a") as af:
            af.write("Training multi-task model on simultaneous age and gender recognition.\n")

        model, best_accs = train_model_multitask(
            model, train_ds, val_ds,
            optimizer=optimizer,
            epochs=cfg.TRAIN.MAX_EPOCHS,
            loss_fn_1=criterions[0],
            loss_fn_2=criterions[1],
            acc_metric_1=eval_metrics[0],
            acc_metric_2=eval_metrics[1],
            log_file=cfg.LOG_FILE,
            tensorboard_dir=cfg.LOG_DIR,
            tensorboard_prefix=cfg.PREFIX,
        )

        return best_accs

        # return test_model_multitask(
        #     model, val_ds,
        #     loss_fn_1=criterions[0],
        #     loss_fn_2=criterions[1],
        #     acc_metric_1=eval_metrics[0],
        #     acc_metric_2=eval_metrics[1],
        # )[1]

    #Run model testing

    