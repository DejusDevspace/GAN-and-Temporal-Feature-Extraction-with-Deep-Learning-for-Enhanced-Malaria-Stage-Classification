# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_size: int):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.img_shape = (3, self.img_size, self.img_size)

        def layer(in_neurons: int, out_neurons: int, normalize_: bool = True, dropout_: bool = False):
            block = [nn.Linear(in_neurons, out_neurons)]
            if normalize_:
                block.append(nn.BatchNorm1d(out_neurons))
            if dropout_:
                block.append(nn.Dropout(0.3))
            block.append(nn.LeakyReLU(0.2))
            return block

        self.model = nn.Sequential(
            *layer(self.latent_dim, 128, normalize_=False, dropout_=True),
            *layer(128, 256, dropout_=True),
            *layer(256, 512, dropout_=True),
            *layer(512, 1024),
            *layer(1024, 2048),
            nn.Linear(2048, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.model(z)
        z = z.view(z.size(0), 3, self.img_size, self.img_size)
        return z
