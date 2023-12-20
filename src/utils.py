from torchtext.transforms import ToTensor

def transfrom_input_AgNews(text, max_length=1024):
    vocab = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    encode = [vocab.index(i)+1 for i in text if i in vocab]
    if len(encode) > max_length:
        encode = encode[:max_length]
    else:
        encode += [0]*(max_length - len(encode))
    encode = ToTensor()(encode)
    return encode[None, :]

def transfrom_input_UITfeedback(text, max_length=1024):
    vocab = list("""AĂÂBCDĐEÊFGHIJKLMNOÔƠPQRSTUƯVWXYZaăâbcdđeêfghijklmnoôơpqrstuưvwxyz0123456789!@#$%^&*()-_+=<>?/.,;:'\"""")
    encode = [vocab.index(i)+1 for i in text if i in vocab]
    if len(encode) > max_length:
        encode = encode[:max_length]
    else:
        encode += [0]*(max_length - len(encode))
    encode = ToTensor()(encode)
    return encode[None, :]
