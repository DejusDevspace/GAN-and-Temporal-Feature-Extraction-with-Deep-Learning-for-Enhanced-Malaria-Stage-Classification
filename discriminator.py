# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_size: int):
        super(Discriminator, self).__init__()

        self.img_size = img_size
        self.img_shape = (3, self.img_size, self.img_size)

        def layer(in_neurons, out_neurons, dropout_=False):
            block = [nn.Linear(in_neurons, out_neurons), nn.LeakyReLU(0.2)]
            if dropout_:
                block.append(nn.Dropout(0.3))
            return block

        self.model = nn.Sequential(
            *layer(int(np.prod(self.img_shape)), 1024, dropout_=True),
            *layer(1024, 512, dropout_=True),
            *layer(512, 256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = z.view(z.size(0), -1)
        return self.model(z)
