import torch
import torch.nn as nn


class Sumformer(nn.Module):
    def __init__(self, vocab_size, max_input_len, emb_dim, e_heads, e_hidden, e_depth, d_heads, d_hidden, d_depth):
        """Text summarisation transformer. Takes as input a document and outputs a summary.

        Args:
            vocab_size (int): Number of tokens in the vocabulary.
            max_input_len (int): Maximum length of the input documents.
            emb_dim (int): Dimensionality of the token embeddings.
            e_heads (int): Number of attention heads in the encoder.
            e_hidden (int): Size of hidden layer in the encoder.
            e_depth (int): Depth of the encoder.
            d_heads (int): Number of attention heads in the decoder.
            d_hidden (int): Size of hidden layer in the decoder.
            d_depth (int): Depth of the decoder.
        """
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim).to(self.device)
        self.pos_emb = nn.Embedding(num_embeddings=max_input_len, embedding_dim=emb_dim).to(self.device)  # * test positional encoding vs. positional embeddings

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(  # * test relu vs gelu & norm_first or not
                d_model=emb_dim, nhead=e_heads, dim_feedforward=e_hidden, dropout=0.1, activation='relu', batch_first=True),
            num_layers=e_depth,
            norm=nn.LayerNorm(emb_dim)).to(self.device)

        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(  # * test relu vs gelu & norm_first or not
                d_model=emb_dim, nhead=d_heads, dim_feedforward=d_hidden, dropout=0.1, activation='relu', batch_first=True),
            num_layers=d_depth,
            norm=nn.LayerNorm(emb_dim)).to(self.device)

        self.toProbs = nn.Linear(in_features=emb_dim, out_features=vocab_size).to(self.device)

    def generate_square_subsequent_mask(self, seq_len):
        """Mask out positions ahead of the current position."""
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def forward(self, docs, sums, doc_mask=None, sum_mask=None):
        docs = docs.to(self.device)
        sums = sums.to(self.device)
        
        if doc_mask is not None:
            doc_mask = doc_mask.to(self.device)
        if sum_mask is not None:
            sum_mask = sum_mask.to(self.device)
        
        docs_emb = self.token_emb(docs) + self.pos_emb(torch.arange(docs.size(1)).unsqueeze(0).to(self.device))
        sums_emb = self.token_emb(sums) + self.pos_emb(torch.arange(sums.size(1)).unsqueeze(0).to(self.device))

        # run the docs through the encoder, ignoring the padding tokens
        docs_enc = self.encoder(docs_emb, src_key_padding_mask=doc_mask)

        # create a mask for subsequent positions
        tgt_mask = self.generate_square_subsequent_mask(sums.size(1))
        # run the sums through the decoder, cross-attending to the encoded docs, ignoring padding tokens and subsequent positions
        sums_dec = self.decoder(sums_emb, docs_enc, tgt_mask=tgt_mask, tgt_key_padding_mask=sum_mask)

        return self.toProbs(sums_dec)

    def generate_greedy_summary(self, doc, tokenizer, max_len=100):
        """Generates a summary using greedy decoding."""        
        # set the model to eval mode
        self.eval()
        
        # tokenise the input and convert to ids tensor
        input_tokens = torch.tensor(tokenizer.encode(doc)).to(self.device)  # TODO implement encode - maybe use tokenizer.encode_plus? - ensure that it returns a tensor of type LongTensor

        # create a mask for the input tokens
        doc_mask = (input_tokens == tokenizer.pad_token_id)  # TODO implement pad_token_id
        doc_mask = doc_mask.masked_fill(doc_mask == 0, float('-inf')).masked_fill(doc_mask == 1, float(0.0))

        # init the output summary with the start token
        output_tokens = torch.tensor([tokenizer.sos_token_id]).to(self.device)  # TODO implement sos_token_id

        # generate until the end token or max_len is reached
        for _ in range(max_len):
            # generate the next token
            output = self.forward(input_tokens, output_tokens, doc_mask)
            next_token = output.argmax(1)[-1]

            # add the next token to the output summary
            output_tokens = torch.cat([output_tokens, next_token.unsqueeze(0)])

            # if the end token is generated, stop
            if next_token == tokenizer.eos_token_id:  # TODO implement eos_token_id
                break
        
        # convert the output summary back to text
        return tokenizer.decode(output_tokens.tolist())  # TODO implement decode

    def generate_beam_summary(self, doc, tokenizer, beam_width=5, max_len=100):  # ! TODO: research beam search
        """Generates a summary using beam search decoding."""
        # set the model to eval mode
        self.eval()

        # tokenise the input and convert to ids tensor
        input_tokens = torch.tensor(tokenizer.encode(doc)).to(self.device)

        # create a mask for the input tokens
        doc_mask = (input_tokens == tokenizer.pad_token_id)
        doc_mask = doc_mask.masked_fill(doc_mask == 0, float('-inf')).masked_fill(doc_mask == 1, float(0.0))

        # init the output summary with the start token
        output_tokens = torch.tensor([tokenizer.sos_token_id]).to(self.device)

        # init the beam
        beam = [(output_tokens, 0.0)]  # start with the start token and a score of 0

        for _ in range(max_len):
            next_beam = []
            for output_tokens, old_score in beam:
                output = self.forward(input_tokens, output_tokens, doc_mask)
                output = torch.nn.functional.log_softmax(output, dim=-1)  # log probabilities for numerical stability
                topk_scores, topk_tokens = output[-1, :].topk(beam_width)  # get the topk for the last token only

                for score, token in zip(topk_scores, topk_tokens):
                    next_output_tokens = torch.cat([output_tokens, token.unsqueeze(0)])
                    next_score = old_score + score.item()
                    next_beam.append((next_output_tokens, next_score))

            # select the top beam_width beams
            beam = sorted(next_beam, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # stop if the top beam ended with the eos_token
            if beam[0][0][-1] == tokenizer.eos_token_id:
                break

        return tokenizer.decode(beam[0][0].tolist())  # TODO implement decode
