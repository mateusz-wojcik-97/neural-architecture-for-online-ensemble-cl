import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from jax import jit, grad, random

config = {
    'enc_size': 512,  # size of latent encoding
    'mem_size': 512,  # number of memory locations (classifiers)
    'k': 25,  # k nearest neighbour lookup parameter
    'vub': 250,  # upper bound for activation function - was 100
    'res': 28,  # resolution - 28 for MNIST & Omniglot
    'col_dims': 1,  # 3 for RGB, 1 for greyscale
    'num_classes': 10,  # number of classes
    'pretrain_n_batches': 10000,  # number of batches in pre-training
    'pretrain_dataset': 'omniglot',
    'main_dataset': 'mnist',
    'batch_size': 60,  # batch size for training main model
    'learning_rate': 1e-4,  # learning rate for training main model
    'weight_decay': 1e-4,  # optimiser weight decay
    'init_scale': 0.1,  # baseline classifier initialiser variance scaling
    'log_every': 10,  # interval for logging accuracies
    'report_every': 500,  # interval for reporting accuracies
    'schedule_type': '10way_split',  # training schedule (defining splits, etc)
    'n_runs': 1,  # number of runs on the schedule
}


class Autoencoder(hk.Module):
    """Autoencoder module."""

    def __init__(self, enc_size: int, res: int, col_dims: int):
        super().__init__()
        self.enc_size = enc_size
        self.res = res
        self.col_dims = col_dims

    def encode(self, image):
        cnn = hk.Sequential(
            [
                hk.Conv2D(output_channels=16, kernel_shape=4, name="enc1"),
                jax.nn.relu,
                hk.Conv2D(output_channels=16, kernel_shape=4, name="enc2"),
                jax.nn.relu,
                hk.Flatten(),
            ]
        )
        mlp1 = hk.Sequential(
            [
                hk.Linear(128, name="enc3"),
                jax.nn.relu,
                hk.Linear(self.enc_size, name="enc4"),
            ]
        )
        mlp2 = hk.Sequential(
            [
                hk.Linear(128, name="enc5"),
                jax.nn.relu,
                hk.Linear(self.enc_size, name="enc6"),
            ]
        )
        feats = cnn(image.reshape([-1, self.res, self.res, self.col_dims]))
        enc_mean = jnp.tanh(mlp1(feats))
        enc_sd = jax.nn.relu(mlp2(feats))
        return (enc_mean, enc_sd)

    def decode(self, latent):
        dcnn = hk.Sequential(
            [
                hk.Linear(128, name="dec1"),
                jax.nn.relu,
                hk.Linear(self.res * self.res * 16, name="dec2"),
                jax.nn.relu,
                hk.Reshape((self.res, self.res, 16)),
                hk.Conv2DTranspose(output_channels=16, kernel_shape=4, name="dec3"),
                jax.nn.relu,
                hk.Conv2DTranspose(
                    output_channels=self.col_dims, kernel_shape=4, name="dec4"
                ),
                jax.nn.sigmoid,
            ]
        )
        image = dcnn(latent).reshape([-1, self.res, self.res, self.col_dims])
        return image

    def forward(self, rng, image):
        (enc_mean, enc_sd) = self.encode(image)
        # Sample
        (rng2, rng) = random.split(rng)
        eps = random.normal(rng2, jnp.shape(enc_mean))
        enc = enc_mean + enc_sd * eps
        image_dec = self.decode(enc)
        out = {
            "enc_mean": enc_mean,
            "enc_sd": enc_sd,
            "image_dec": image_dec,
        }
        return out


def encoder(rng, image):
    autoencoder = Autoencoder(config['enc_size'], res=28, col_dims=1)
    out = autoencoder.forward(rng, image)
    return out


def kl_divergence(mean, sd):
    kl = -0.5 * (1.0 + jnp.log(sd ** 2) - mean ** 2 - sd ** 2)
    return kl


def autoencoder_losses(enc_params, rng, images):
    encoder_net = hk.transform(encoder)
    (rng2, rng) = random.split(rng)
    autoencoder_out = encoder_net.apply(enc_params, rng, rng2, images)
    enc_means = autoencoder_out["enc_mean"]
    enc_sds = autoencoder_out["enc_sd"]
    image_decs = autoencoder_out["image_dec"]
    # Decoder reconstruction loss
    decoder_loss = jnp.mean((images - image_decs) ** 2)
    # Decoder KL loss
    kld = kl_divergence(enc_means, enc_sds + 1e-10)  # add epsilon to avoid sd=0
    kl_loss = jnp.mean(kld)
    # Total loss
    beta = 0.001  # weighting of KL term
    tot_loss = decoder_loss + beta * kl_loss
    losses = {
        "tot_loss": tot_loss,
        "decoder_loss": decoder_loss,
        "kl_loss": kl_loss,
    }
    return losses


def make_encoder_optimiser():
    opt = optax.adam(learning_rate=0.001)  # learning rate was 0.001
    return opt


def autoencoder_loss(enc_params, rng, images):
    losses = autoencoder_losses(enc_params, rng, images)
    return losses["tot_loss"]


@jit
def update_autoencoder(enc_params, rng, opt_state, images):
    opt = make_encoder_optimiser()
    grads = grad(autoencoder_loss)(enc_params, rng, images)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(enc_params, updates)
    return new_params, opt_state


def get_batch(dataset, dataset_name):
    batch = next(dataset)
    batch_size = batch["image"].shape[0]
    images = batch["image"]
    if dataset_name == "omniglot":
        images = tf.image.resize(images, [config["res"], config["res"]])
        images = images[:, :, :, 0]
    images = tf.reshape(images, [batch_size, 28, 28, 1]) / 255
    if dataset_name == "omniglot":
        images = 1 - images  # raw Omniglot characters are white (1) on black (0)
    labels = batch["label"]
    one_hots = tf.one_hot(batch["label"], config["num_classes"])
    return images.numpy(), one_hots.numpy(), labels.numpy()


def get_dataset(dataset_name, train_or_test, batch_size, filter_labels=None):
    filter_fn = lambda batch: tf.reduce_any(tf.equal(batch["label"], filter_labels))
    dataset = tfds.load(dataset_name, split=train_or_test, as_supervised=False)
    if filter_labels is not None:
        dataset = dataset.filter(filter_fn)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = iter(dataset)
    return dataset


def get_encoder():
    encoder_net = hk.transform(encoder)
    return encoder_net


def initialise_encoder(rng):
    """Initialise internal encoder for pre-training."""
    encoder_net = get_encoder()
    # Get dummy batch
    batch_size = 24
    train_set = get_dataset(config["pretrain_dataset"], "train", batch_size)
    batch = get_batch(train_set, config["pretrain_dataset"])
    (images, _, _) = batch
    (rng2, rng) = random.split(rng)
    (rng3, rng) = random.split(rng)
    enc_params = encoder_net.init(rng2, rng3, images)
    return enc_params


@jit
def apply_encoder(enc_params, rng, images):
    encoder_net = get_encoder()
    (rng2, rng) = random.split(rng)
    encoder_out = encoder_net.apply(enc_params, rng, rng2, images)
    enc = encoder_out["enc_mean"]
    return enc


if __name__ == "__main__":
    pass
