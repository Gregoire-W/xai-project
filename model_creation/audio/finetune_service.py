# maths
import matplotlib.pyplot as plt

# deep learning
import tensorflow as tf


def create_datasets(train_path, val_path, test_path, batch_size, img_size):
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_path,
        shuffle=True,
        batch_size=batch_size,
        image_size=img_size
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        val_path,
        shuffle=True,
        batch_size=batch_size,
        image_size=img_size
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_path,
        shuffle=False,
        batch_size=batch_size,
        image_size=img_size
    )

    AUTOTUNE = tf.data.AUTOTUNE
    
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset


def create_model(base_model, pre_process, dropout):
    
    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    inputs = tf.keras.Input(shape=(224, 224, 3), name="img_input_finetune")
    x = pre_process(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model, global_average_layer, prediction_layer

def train_classifier(model, lr, epochs, train_dataset, validation_dataset, cp_path):
    base_learning_rate = lr
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC()
        ],
    )

    initial_epochs = epochs
    
    loss0, accuracy0, auc0 = model.evaluate(validation_dataset)

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))
    print("initial auc: {:.2f}".format(auc0))
        
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(cp_path / "model_epoch_{epoch:02d}.keras"),
        save_weights_only=False,
        save_freq="epoch",
        verbose=0
    )

    history = model.fit(
        train_dataset,
        epochs=initial_epochs,
        validation_data=validation_dataset,
        callbacks=[checkpoint_cb]
    )

    return model, history

def plot_metrics(history, acc, val_acc, loss, val_loss, auc, val_auc):
    auc_keys = [elem for elem in history.history.keys() if "auc" in elem]
    val_auc_key = [elem for elem in auc_keys if elem.startswith("val")][0]
    auc_key = [elem for elem in auc_keys if not elem.startswith("val")][0]
    
    acc += history.history["accuracy"]
    val_acc += history.history["val_accuracy"]
    auc += history.history[auc_key] 
    
    loss += history.history["loss"]
    val_loss += history.history["val_loss"]
    val_auc += history.history[val_auc_key]
    
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy")
    plt.ylim([min(plt.ylim()),1])
    plt.title("Training and Validation Accuracy")
    
    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel("Cross Entropy")
    plt.ylim([0,1.0])
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.show()

    return acc, val_acc, loss, val_loss, auc, val_auc

def get_best_cp(cp_path, test_dataset, validation_dataset):
    results = []
    for model_path in sorted(cp_path.iterdir()):
        checkpoint = tf.keras.models.load_model(model_path)
        
        # Test accuracy
        _, test_acc, _ = checkpoint.evaluate(test_dataset, verbose=0)
        
        # Validation accuracy
        _, val_accuracy, _ = checkpoint.evaluate(validation_dataset, verbose=0)
        
        results.append({
            "model_path": model_path,
            "epoch": model_path.name,
            "test_acc": test_acc,
            "val_accuracy": val_accuracy,
            "gap": val_accuracy - test_acc
        })
        
        print(f"{model_path.name:25} | Test: {test_acc:.4f} | Val: {val_accuracy:.4f} | Gap: {val_accuracy - test_acc:.4f}")
    
    best = max(results, key=lambda x: x['test_acc'])
    print(f"\nBest checkpoint: {best['epoch']} with test accuracy = {best['test_acc']:.4f}")

    return tf.keras.models.load_model(best["model_path"])

def fine_tune_classifier(model, base_model, train_dataset, validation_dataset, fine_tune_at, lr, cp_path, initial_epochs, epochs):
    base_model.trainable = True

    print("Number of layers in the base model: ", len(base_model.layers))
    
    # Freeze all the layers before the 'fine_tune_at' layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable = False

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC()
        ],
    )
    print("Number of trainable variables in the model: ", len(model.trainable_variables))
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(cp_path / "model_epoch_{epoch:02d}.keras"),
        save_weights_only=False,
        save_freq="epoch",
        verbose=0
    )

    fine_tune_epochs = epochs
    total_epochs =  initial_epochs + fine_tune_epochs
    
    
    history_fine = model.fit(
        train_dataset,
        epochs=total_epochs,
        initial_epoch=initial_epochs,
        validation_data=validation_dataset,
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
    acc, val_acc, loss, val_loss, auc, val_auc = [], [], [], [], [], []
    classifier_cp_path = cp_path / "classifier"
    fine_tuned_cp_path = cp_path / "fine_tuned"

    # check path exists
    if not classifier_cp_path.exists():
        classifier_cp_path.mkdir(parents=True, exist_ok=True)
    if not fine_tuned_cp_path.exists():
        fine_tuned_cp_path.mkdir(parents=True, exist_ok=True)
    if not model_save_path.exists():
        model_save_path.mkdir(parents=True, exist_ok=True)
    
    train_dataset, validation_dataset, test_dataset = create_datasets(train_path, val_path, test_path, batch_size, img_size)

    model, global_average_layer, prediction_layer = create_model(base_model, pre_process, dropout)

    print("Start training classifier")
    model, history = train_classifier(model, lr_classifier, epochs_classifier, train_dataset, validation_dataset, classifier_cp_path)
    print(f"metrics: {history.history.keys()}")

    acc, val_acc, loss, val_loss, auc, val_auc = plot_metrics(history, acc, val_acc, loss, val_loss, auc, val_auc)

    print("Let's select the best checkpoint based on accuracy:")
    model = get_best_cp(classifier_cp_path, test_dataset, validation_dataset)
    model.save(str(model_save_path / "classifier.keras"))
    base_model = model.get_layer(base_model_layer_name)

    print("Start finetuning classifier")
    model, history = fine_tune_classifier(model, base_model, train_dataset, validation_dataset, fine_tune_at, lr_finetune, fine_tuned_cp_path, epochs_classifier, epochs_finetune)

    acc, val_acc, loss, val_loss, auc, val_auc = plot_metrics(history, acc, val_acc, loss, val_loss, auc, val_auc)

    print("Let's select the best checkpoint based on accuracy:")
    model = get_best_cp(fine_tuned_cp_path, test_dataset, validation_dataset)
    model.save(str(model_save_path / "fine_tuned.keras"))

    grad_cam = create_grad_cam_model(last_conv_layer_name, base_model, global_average_layer, prediction_layer)
    grad_cam.save(str(model_save_path / "grad_cam.keras"))