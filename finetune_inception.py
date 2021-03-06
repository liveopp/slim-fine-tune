import argparse
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='food-data/train')
parser.add_argument('--val_dir', default='food-data/val')
parser.add_argument('--log_dir', default='inception_log')
parser.add_argument('--model_path', default='inception_v3.ckpt', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=2, type=int)
parser.add_argument('--num_epochs2', default=2, type=int)
parser.add_argument('--learning_rate1', default=1e-1, type=float)
parser.add_argument('--learning_rate2', default=1e-3, type=float)
parser.add_argument('--dropout_keep_prob', default=0.6, type=float)
parser.add_argument('--batch_decay', default=0.99, type=float)

IMG_HEIGHT, IMG_WIDTH = 299, 299


def list_images(directory):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    labels = os.listdir(directory)
    # Sort the labels so that training and validation get them in the same order
    labels.sort()

    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            if 'jpg' in f.lower() or 'png' in f.lower():
                files_and_labels.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = list(set(labels))

    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i

    labels = [label_to_int[l] for l in labels]

    return filenames, labels


def check_accuracy(sess, correct_prediction, is_training, dataset_init_op):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction, {is_training: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc


def main(args):
    # Get the list of filenames and corresponding list of labels for training et validation
    train_filenames, train_labels = list_images(args.train_dir)
    val_filenames, val_labels = list_images(args.val_dir)
    assert set(train_labels) == set(val_labels),\
           "Train and val labels don't correspond:\n{}\n{}".format(set(train_labels),
                                                                   set(val_labels))

    num_classes = len(set(train_labels))

    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computation graph with all the
    # necessary operations: loss, training op, accuracy...
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    graph = tf.Graph()
    with graph.as_default():
        # Standard preprocessing for VGG on ImageNet taken from here:
        # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
        # Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf

        # Preprocessing (for both training and validation):
        # (1) Decode the image from jpg format
        # (2) Resize the image so its smaller side is 256 pixels long
        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            image = tf.image.decode_jpeg(image_string, channels=3)          # (1)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)

            smallest_side = 300.0
            height, width = tf.shape(image)[0], tf.shape(image)[1]
            height = tf.to_float(height)
            width = tf.to_float(width)

            scale = tf.cond(tf.greater(height, width),
                            lambda: smallest_side / width,
                            lambda: smallest_side / height)
            new_height = tf.to_int32(height * scale)
            new_width = tf.to_int32(width * scale)

            resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
            resized_image = 2 * (resized_image - 0.5)
            return resized_image, label

        def training_preprocess(image, label):
            crop_image = tf.random_crop(image, [IMG_HEIGHT, IMG_WIDTH, 3])
            flip_image = tf.image.random_flip_left_right(crop_image)                # (4)
            centered_image = tf.image.random_flip_up_down(flip_image)
            return centered_image, label

        def val_preprocess(image, label):
            centered_image = tf.image.resize_image_with_crop_or_pad(image, IMG_HEIGHT, IMG_WIDTH)
            return centered_image, label

        # Training dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
        train_dataset = train_dataset.shuffle(buffer_size=len(train_filenames))
        train_dataset = train_dataset.map(_parse_function, num_parallel_calls=args.num_workers)
        train_dataset = train_dataset.map(training_preprocess, num_parallel_calls=args.num_workers)
        batched_train_dataset = train_dataset.batch(args.batch_size)

        # Validation dataset
        val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
        val_dataset = val_dataset.map(_parse_function,
            num_parallel_calls=args.num_workers)
        val_dataset = val_dataset.map(val_preprocess,
            num_parallel_calls=args.num_workers)
        batched_val_dataset = val_dataset.batch(args.batch_size)


        # Now we define an iterator that can operator on either dataset.
        # The iterator can be reinitialized by calling:
        #     - sess.run(train_init_op) for 1 epoch on the training set
        #     - sess.run(val_init_op)   for 1 epoch on the valiation set
        # Once this is done, we don't need to feed any value for images and labels
        # as they are automatically pulled out from the iterator queues.

        # A reinitializable iterator is defined by its structure. We could use the
        # `output_types` and `output_shapes` properties of either `train_dataset`
        # or `validation_dataset` here, because they are compatible.
        iterator = tf.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                           batched_train_dataset.output_shapes)
        images, labels = iterator.get_next()

        train_init_op = iterator.make_initializer(batched_train_dataset)
        val_init_op = iterator.make_initializer(batched_val_dataset)

        # Indicates whether we are in training or in test mode
        is_training = tf.placeholder(tf.bool)

        # ---------------------------------------------------------------------
        # Now that we have set up the data, it's time to set up the model.
        # For this example, we'll use VGG-16 pretrained on ImageNet. We will remove the
        # last fully connected layer (fc8) and replace it with our own, with an
        # output size num_classes=8
        # We will first train the last layer for a few epochs.
        # Then we will train the entire model on our dataset for a few epochs.

        # Get the pretrained model, specifying the num_classes argument to create a new
        # fully connected replacing the last one, called "vgg_16/fc8"
        # Each model has a different architecture, so "vgg_16/fc8" will change in another model.
        # Here, logits gives us directly the predicted scores we wanted from the images.
        # We pass a scope to initialize "vgg_16/fc8" weights with he_initializer
        inception = nets.inception
        with slim.arg_scope(inception.inception_v3_arg_scope(
            batch_norm_decay=args.batch_decay)):
            logits, _ = inception.inception_v3(images, num_classes=num_classes,
                    is_training=is_training, dropout_keep_prob=args.dropout_keep_prob)

        # Specify where the model checkpoint is (pretrained weights).
        model_path = args.model_path
        assert(os.path.isfile(model_path))

        # Restore only the layers up to fc7 (included)
        # Calling function `init_fn(sess)` will load all the pretrained weights.
        exclude = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        init_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore)

        # Initialization operation from scratch for the new "fc8" layers
        # `get_variables` will only return the variables whose name starts with the given pattern
        tuning_variables = []
        for v in exclude:
            tuning_variables += slim.get_variables(v)
        fc8_init = tf.variables_initializer(tuning_variables)

        # ---------------------------------------------------------------------
        # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
        # We can then call the total loss easily
        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss = tf.losses.get_total_loss()

        global_step = tf.train.get_or_create_global_step()
        learning_rate1 = tf.train.exponential_decay(learning_rate=args.learning_rate1,
                global_step=global_step,
                decay_steps=100, decay_rate=0.5)
        learning_rate2 = tf.train.exponential_decay(learning_rate=args.learning_rate2,
                global_step=global_step,
                decay_steps=100, decay_rate=0.2)
        # First we want to train only the reinitialized last layer fc8 for a few epochs.
        # We run minimize the loss only with respect to the fc8 variables (weight and bias).
        fc8_optimizer = tf.train.MomentumOptimizer(learning_rate1, 0.8, use_nesterov=True)

        # Then we want to finetune the entire model for a few epochs.
        # We run minimize the loss only with respect to all the variables.
        full_optimizer = tf.train.MomentumOptimizer(learning_rate2, 0.8, use_nesterov=True)
        #full_train_op = slim.learning.create_train_op(loss, full_optimizer)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            fc8_train_op = fc8_optimizer.minimize(loss, global_step,
                    var_list=tuning_variables)
            full_train_op = full_optimizer.minimize(loss, global_step)
        opt_init = [var.initializer for var in tf.global_variables() if 'Momentum' in var.name]
        step_init = global_step.initializer

        # Evaluation metrics
        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
        accuracy_top_5 = tf.reduce_mean(tf.to_float(tf.nn.in_top_k(predictions=logits, targets=labels, k=5)))

        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('accuracy_top_5', accuracy_top_5)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(args.log_dir + '/train', graph)
        eval_writer = tf.summary.FileWriter(args.log_dir + '/eval')

        tf.get_default_graph().finalize()

    # --------------------------------------------------------------------------
    # Now that we have built the graph and finalized it, we define the session.
    # The session is the interface to *run* the computational graph.
    # We can call our training operations with `sess.run(train_op)` for instance
    with tf.Session(graph=graph) as sess:
        init_fn(sess)  # load the pretrained weights
        sess.run(fc8_init)  # initialize the new fc8 layer
        sess.run(opt_init)

        # Update only the last layer for a few epochs.
        step = 0
        sess.run(step_init)
        for epoch in range(args.num_epochs1):
            # Run an epoch over the training data.
            print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
            # Here we initialize the iterator with the training set.
            # This means that we can go through an entire epoch until the iterator becomes empty.
            sess.run(train_init_op)
            while True:
                try:
                    step += 1
                    acc, summary, _ = sess.run([accuracy, merged,
                        fc8_train_op], {is_training: True})
                    train_writer.add_summary(summary, step)
                    if step % 100 == 0:
                        print(f'step: {step} train accuracy: {acc}')
                except tf.errors.OutOfRangeError:
                    break

            # Check accuracy on the train and val sets every epoch.
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
            print('Val accuracy: %f\n' % val_acc)


        # Train the entire model for a few more epochs, continuing with the *same* weights.
        sess.run(step_init)
        for epoch in range(args.num_epochs2):
            print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
            sess.run(train_init_op)
            while True:
                try:
                    step += 1
                    acc, summary, _ = sess.run([accuracy, merged,
                        full_train_op], {is_training: True})
                    train_writer.add_summary(summary, step)
                    if step % 100 == 0:
                        print(f'step: {step} train accuracy: {acc}')
                except tf.errors.OutOfRangeError:
                    break

            # Check accuracy on the train and val sets every epoch
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
            print('Val accuracy: %f\n' % val_acc)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
