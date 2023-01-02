import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers=1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Linear(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, source):
        embedded = self.embedding(source)
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers=1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.rnn = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=n_layers, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        output, (hidden, cell) = self.rnn(inputs)
        prediction = self.fc_out(output)
        return prediction, (hidden, cell)


class VariantionalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_hidden=0) -> None:
        super().__init__()

        self.n_hidden = n_hidden

        layers = nn.ModuleList([])
        layers += [nn.Linear(input_dim, hidden_dim)]
        layers += [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.n_hidden)]
        layers += [nn.Linear(hidden_dim, latent_dim), nn.Linear(hidden_dim, latent_dim)]
        self.enc_layers = layers

        layers = nn.ModuleList([])
        layers += [nn.Linear(latent_dim, hidden_dim)]
        layers += [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.n_hidden)]
        layers += [nn.Linear(hidden_dim, input_dim)]
        self.dec_layers = layers

        self.activation = nn.ReLU()

    def encode(self, inputs):
        hidden = self.activation(self.enc_layers[0](inputs))
        for i in range(self.n_hidden):
            hidden = self.activation(self.enc_layers[i + 1](hidden))
        mean = self.enc_layers[-2](hidden)
        logvar = self.enc_layers[-1](hidden)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mean + eps * std
        return latent

    def decode(self, latent) -> torch.Tensor:
        hidden = self.activation(self.dec_layers[0](latent))
        for i in range(self.n_hidden):
            hidden = self.activation(self.dec_layers[i + 1](hidden))
        reconst = self.dec_layers[-1](hidden)
        return reconst

    def forward(self, inputs):
        mean, logvar = self.encode(inputs)
        latent = self.reparameterization(mean, logvar)
        reconst = self.decode(latent)
        return reconst, mean, logvar


class Seq2SeqMelodyComposer(nn.Module):
    def __init__(self, config, device) -> None:
        super().__init__()

        self.encoder = Encoder(
            config.model.encoder.input_dim,
            config.model.encoder.emb_dim,
            config.model.encoder.hidden_dim,
            config.model.encoder.n_layers,
        ).to(device)
        self.decoder = Decoder(
            config.model.decoder.output_dim,
            config.model.decoder.hidden_dim,
            config.model.decoder.n_layers,
        ).to(device)
        self.vae = VariantionalAutoEncoder(
            config.model.encoder.hidden_dim,
            config.model.vae.hidden_dim,
            config.model.vae.latent_dim,
            config.model.vae.n_hidden,
        ).to(device)

    def forward(self, inputs) -> torch.Tensor:
        seq_len = inputs.shape[1]
        _, encoder_state = self.encoder(inputs)
        hiddens = torch.squeeze(encoder_state[0])
        hiddens = hiddens.unsqueeze(0)
        reconst_state, _, _ = self.vae(hiddens)
        inputs = reconst_state.unsqueeze(1)
        inputs = inputs.repeat(1, seq_len, 1)
        outputs, _ = self.decoder(inputs)
        return outputs


def get_model(config, device):
    model = Seq2SeqMelodyComposer(config, device)
    return model
