# maths
import matplotlib.pyplot as plt
import math

# data
import pandas as pd
from sklearn.metrics import roc_auc_score

# deep learning
import tensorflow as tf


def create_generators(train_path, valid_path, test_path, batch_size, img_size):

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)
    
    pathologies = train_df.columns[5:]
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="Path",
        y_col=pathologies,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="raw",
        shuffle=True
    )
    
    validation_generator = datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col="Path",
        y_col=pathologies,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="raw",
        shuffle=False
    )

    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="Path",
        y_col=pathologies,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="raw",
        shuffle=False
    )

    return train_generator, validation_generator, test_generator, pathologies


def create_model(base_model, pre_process, dropout, num_labels):
    
    base_model.trainable = False

    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip("horizontal"),
      tf.keras.layers.RandomRotation(0.05)
    ])
    
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    prediction_layer = tf.keras.layers.Dense(num_labels, activation="sigmoid")

    inputs = tf.keras.Input(shape=(224, 224, 3), name="img_input_finetune")
    x = data_augmentation(inputs)
    x = pre_process(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = prediction_layer(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model, global_average_layer, prediction_layer

def train_classifier(model, lr, epochs, train_generator, validation_generator, cp_path):
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(multi_label=True, name="auc")]
    )

    VAL_STEPS = math.ceil(validation_generator.n / validation_generator.batch_size)
    loss0, auc0 = model.evaluate(
        validation_generator,
        steps=VAL_STEPS,
        verbose=1
    )

    print("initial loss: {:.2f}".format(loss0))
    print("initial auc: {:.2f}".format(auc0))
        
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(cp_path / "model_epoch_{epoch:02d}.keras"),
        save_weights_only=False,
        save_freq="epoch",
        verbose=0
    )

    TRAIN_STEPS = math.ceil(train_generator.n / train_generator.batch_size)
    history = model.fit(
        train_generator,
        steps_per_epoch=TRAIN_STEPS,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=VAL_STEPS,
        callbacks=[checkpoint_cb]
    )

    return model, history

def plot_metrics(history, loss, val_loss, auc, val_auc):
    
    auc += history.history["auc"]
    val_auc += history.history["val_auc"]
    
    loss += history.history["loss"]
    val_loss += history.history["val_loss"]
    
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(auc, label="Training Multi Label AUC")
    plt.plot(val_auc, label="Validation Multi Label AUC")
    plt.legend(loc="lower right")
    plt.ylabel("Multi Label AUC")
    plt.ylim([min(plt.ylim()),1])
    plt.title("Training and Validation Multi Label AUC")
    
    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel("Cross Entropy")
    plt.ylim([0,1.0])
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.show()

    return loss, val_loss, auc, val_auc

def get_best_cp(cp_path, test_generator, validation_generator):
    results = []
    for model_path in sorted(cp_path.iterdir()):
        checkpoint = tf.keras.models.load_model(model_path)
        
        # Test multi label auc
        TEST_STEPS = math.ceil(test_generator.n / test_generator.batch_size)
        _, test_auc = checkpoint.evaluate(
            test_generator,
            steps=TEST_STEPS,
            verbose=0
        )
        
        # Validation multi label auc
        VAL_STEPS = math.ceil(validation_generator.n / validation_generator.batch_size)
        _, val_auc = checkpoint.evaluate(
            validation_generator,
            steps=VAL_STEPS,
            verbose=0
        )
        
        results.append({
            "model_path": model_path,
            "epoch": model_path.name,
            "test_auc": test_auc,
            "val_auc": val_auc,
            "gap": val_auc - test_auc
        })
        
        print(f"{model_path.name:25} | Test: {test_auc:.4f} | Val: {val_auc:.4f} | Gap: {val_auc - test_auc:.4f}")
    
    best = max(results, key=lambda x: x["test_auc"])
    print(f"\nBest checkpoint: {best["epoch"]} with test multi label auc = {best["test_auc"]:.4f}")

    return tf.keras.models.load_model(best["model_path"])

def fine_tune_classifier(model, base_model, train_generator, validation_generator, fine_tune_at, lr, cp_path, initial_epochs, epochs):
    base_model.trainable = True

    print("Number of layers in the base model: ", len(base_model.layers))
    
    # Freeze all the layers before the 'fine_tune_at' layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable = False
    print("Number of trainable variables in the model: ", len(model.trainable_variables))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(multi_label=True, name="auc")]
    )
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(cp_path / "model_epoch_{epoch:02d}.keras"),
        save_weights_only=False,
        save_freq="epoch",
        verbose=0
    )

    total_epochs =  initial_epochs + epochs 

    TRAIN_STEPS = math.ceil(train_generator.n / train_generator.batch_size)
    VAL_STEPS = math.ceil(validation_generator.n / validation_generator.batch_size)
    history_fine = model.fit(
        train_generator,
        steps_per_epoch=TRAIN_STEPS,
        epochs=total_epochs,
        initial_epoch=initial_epochs,
        validation_data=validation_generator,
        validation_steps=VAL_STEPS,
        callbacks=[checkpoint_cb]
    )

    return model, history_fine

def create_grad_cam_model(last_conv_layer_name, base_model, global_average_layer, prediction_layer):

    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    
    base_model_output = base_model.output
    x = global_average_layer(base_model_output)

    final_predictions = prediction_layer(x)
    
    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, final_predictions]
    )
    
    return grad_model

def fine_tune(
    train_path,
    val_path,
    test_path,
    batch_size,
    img_size,
    base_model,
    base_model_layer_name,
    pre_process,
    dropout,
    lr_classifier,
    epochs_classifier,
    cp_path,
    fine_tune_at,
    lr_finetune,
    epochs_finetune,
    model_save_path,
    last_conv_layer_name,
):
    loss, val_loss, auc, val_auc = [], [], [], []
    classifier_cp_path = cp_path / "classifier"
    fine_tuned_cp_path = cp_path / "fine_tuned"

    # check path exists
    if not classifier_cp_path.exists():
        classifier_cp_path.mkdir(parents=True, exist_ok=True)
    if not fine_tuned_cp_path.exists():
        fine_tuned_cp_path.mkdir(parents=True, exist_ok=True)
    if not model_save_path.exists():
        model_save_path.mkdir(parents=True, exist_ok=True)
    
    train_generator, validation_generator, test_generator, pathologies = create_generators(train_path, val_path, test_path, batch_size, img_size)

    model, global_average_layer, prediction_layer = create_model(base_model, pre_process, dropout, train_generator.labels.shape[1])

    print("Start training classifier")
    model, history = train_classifier(model, lr_classifier, epochs_classifier, train_generator, validation_generator, classifier_cp_path)
    print(f"metrics: {history.history.keys()}")

    loss, val_loss, auc, val_auc = plot_metrics(history, loss, val_loss, auc, val_auc)

    print("Let's select the best checkpoint based on multi label auc:")
    model = get_best_cp(classifier_cp_path, test_generator, validation_generator)
    model.save(str(model_save_path / "classifier.keras"))
    base_model = model.get_layer(base_model_layer_name)

    print("Start finetuning classifier")
    model, history = fine_tune_classifier(model, base_model, train_generator, validation_generator, fine_tune_at, lr_finetune, fine_tuned_cp_path, epochs_classifier, epochs_finetune)

    loss, val_loss, auc, val_auc = plot_metrics(history, loss, val_loss, auc, val_auc)

    print("Let's select the best checkpoint based on multi label auc:")
    model = get_best_cp(fine_tuned_cp_path, test_generator, validation_generator)
    model.save(str(model_save_path / "fine_tuned.keras"))

    y_pred = model.predict(test_generator)
    y_true = test_generator.labels
    print("Individual auc for each pathology:")
    for i, label in enumerate(pathologies):
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        print(f"{label}: AUC = {auc:.4f}")
    
    grad_cam = create_grad_cam_model(last_conv_layer_name, base_model, global_average_layer, prediction_layer)
    grad_cam.save(str(model_save_path / "grad_cam.keras"))