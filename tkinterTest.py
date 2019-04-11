import tkinter as tk


class args(object):
    init = ""


window = tk.Tk()
window.title('Spammer Detection')
window.geometry('1280x720')


def setHyperPrams(window, text, x, y):
    l = tk.Label(window,
                 text=text,    # 标签的文字
                 font=('Arial', 12),     # 字体和字体大小 # 标签长宽
                 )
    # l.pack(side='top')
    l.place(x=x, y=y, anchor='nw')
    e = tk.Entry(window)
    e.place(x=(x+250), y=y, anchor="nw")
    # e.pack(side='top')
    return l, e


def setRadioButtons(var, label, texts, values, command, x, y):

    l = tk.Label(window,
                 text=label,  
                 font=('Arial', 12),   
                 )
    l.place(x=x, y=y, anchor='nw')

    for t, v in zip(texts, values):
        y = y+40
        tk.Radiobutton(window, text=t, font=('Arial', 12),
                       variable=var, value=v,
                       command=command,).place(x=x, y=y, anchor='nw')


############### Dataset ###############

def Take_DatasetSelection():
    print("Selected Dataset: ", datasetSelected.get())


datasetSelected = tk.StringVar()

setRadioButtons(datasetSelected, "Dataset:",
                ["HSpam14Dataset", "HoneypotDataset"], ['HSpam14', 'Honeypot'],
                Take_DatasetSelection, 10, 10)


############### Models ###############


def Take_ModelSelection():
    print("Selected Model: ", modelSelected.get())
    print("Calling here")
    printHyperParamsSetting(window, modelSelected.get(), 250, 10)


modelSelected = tk.StringVar()

models_list = ['SSCL', 'GatedCNN', 'SelfAttn']

setRadioButtons(modelSelected, "Model:", models_list,
                models_list, Take_ModelSelection, 10, 150)


x = 250
y = 10


GatedCNN_embedingDim_label, GatedCNN_embedingDim = setHyperPrams(
    window, "GatedCNN_embedingDim", x, y)
GatedCNN_convDim_label, GatedCNN_convDim = setHyperPrams(
    window, "GatedCNN_convDim", x, y+(30*1))
GatedCNN_kernel_label, GatedCNN_kernel = setHyperPrams(
    window, "GatedCNN_kernel", x, y+(30*2))
GatedCNN_stride_label, GatedCNN_stride = setHyperPrams(
    window, "GatedCNN_stride", x, y+(30*3))
GatedCNN_pad_label, GatedCNN_pad = setHyperPrams(
    window, "GatedCNN_pad", x, y+(30*4))
GatedCNN_layers_label, GatedCNN_layers = setHyperPrams(
    window, "GatedCNN_layers", x, y+(30*5))
GatedCNN_dropout_label, GatedCNN_dropout = setHyperPrams(
    window, "GatedCNN_dropout", x, y+(30*6))

SSCL_RNNHidden_label, SSCL_RNNHidden = setHyperPrams(
    window, "SSCL_RNNHidden", x, y)
SSCL_CNNDim_label, SSCL_CNNDim = setHyperPrams(
    window, "SSCL_CNNDim", x, y+(30*1))
SSCL_CNNKernel_label, SSCL_CNNKernel = setHyperPrams(
    window, "SSCL_CNNKernel", x, y+(30*2))
SSCL_CNNDropout_label, SSCL_CNNDropout = setHyperPrams(
    window, "SSCL_CNNDropout", x, y+(30*3))
SSCL_LSTMDropout_label, SSCL_LSTMDropout = setHyperPrams(
    window, "SSCL_LSTMDropout", x, y+(30*4))
SSCL_LSTMLayers_label, SSCL_LSTMLayers = setHyperPrams(
    window, "SSCL_LSTMLayers", x, y+(30*5))

SelfAttn_LenMaxSeq_label, SelfAttn_LenMaxSeq = setHyperPrams(
    window, "SelfAttn_LenMaxSeq", x, y)
SelfAttn_ModelDim_label, SelfAttn_ModelDim = setHyperPrams(
    window, "SelfAttn_ModelDim", x, y+(30*1))
SelfAttn_FFInnerDim_label, SelfAttn_FFInnerDim = setHyperPrams(
    window, "SelfAttn_FFInnerDim", x, y+(30*2))
SelfAttn_NumLayers_label, SelfAttn_NumLayers = setHyperPrams(
    window, "SelfAttn_NumLayers", x, y+(30*3))
SelfAttn_NumHead_label, SelfAttn_NumHead = setHyperPrams(
    window, "SelfAttn_NumHead", x, y+(30*4))
SelfAttn_KDim_label, SelfAttn_KDim = setHyperPrams(window, "SelfAttn_KDim", x, y+(30*5))
SelfAttn_VDim_label, SelfAttn_VDim = setHyperPrams(window, "SelfAttn_VDim", x, y+(30*6))
SelfAttn_Dropout_label, SelfAttn_Dropout = setHyperPrams(
    window, "SelfAttn_Dropout", x, y+(30*7))

GatedCNN_label = [
    GatedCNN_embedingDim_label,
    GatedCNN_convDim_label,
    GatedCNN_kernel_label,
    GatedCNN_stride_label,
    GatedCNN_pad_label,
    GatedCNN_layers_label,
    GatedCNN_dropout_label,
]

GatedCNN_entry = [
    GatedCNN_embedingDim,
    GatedCNN_convDim,
    GatedCNN_kernel,
    GatedCNN_stride,
    GatedCNN_pad,
    GatedCNN_layers,
    GatedCNN_dropout,
]

SSCL_label = [
    SSCL_RNNHidden_label,
    SSCL_CNNDim_label,
    SSCL_CNNKernel_label,
    SSCL_CNNDropout_label,
    SSCL_LSTMDropout_label,
    SSCL_LSTMLayers_label,

]

SSCL_entry = [
    
    SSCL_RNNHidden,
    SSCL_CNNDim,
    SSCL_CNNKernel,
    SSCL_CNNDropout,
    SSCL_LSTMDropout,
    SSCL_LSTMLayers,
]


SelfAttn_label = [
    SelfAttn_LenMaxSeq_label,
    SelfAttn_ModelDim_label,
    SelfAttn_FFInnerDim_label,
    SelfAttn_NumLayers_label,
    SelfAttn_NumHead_label,
    SelfAttn_KDim_label,
    SelfAttn_VDim_label,
    SelfAttn_Dropout_label,
]

SelfAttn_entry = [
    
    SelfAttn_LenMaxSeq,
    SelfAttn_ModelDim,
    SelfAttn_FFInnerDim,
    SelfAttn_NumLayers,
    SelfAttn_NumHead,
    SelfAttn_KDim,
    SelfAttn_VDim,
    SelfAttn_Dropout,
]


def forgetLabelAndEntry(label, entry):
    for l, e in zip(label, entry):
        l.place_forget()
        e.place_forget()


forgetLabelAndEntry(SSCL_label, SSCL_entry)
forgetLabelAndEntry(SelfAttn_label, SelfAttn_entry)
forgetLabelAndEntry(GatedCNN_label, GatedCNN_entry)

def placeLabelAndEntry(labels, entries, x, y):
    for l, e in zip(labels, entries):
        y = y + 30
        l.place(x=x, y=y, anchor='nw')
        e.place(x=(x+250), y=y, anchor="nw")


def printHyperParamsSetting(window, model, x, y):

    print('This method get called')

    print("Model Selected is: ", model)

    x = 250
    y = 10

    if model == 'GatedCNN':

        print("GatedCNN get called")

        forgetLabelAndEntry(SSCL_label, SSCL_entry)
        forgetLabelAndEntry(SelfAttn_label, SelfAttn_entry)
        placeLabelAndEntry(GatedCNN_label, GatedCNN_entry, x, y)

    elif model == "SSCL":

        print("SSCL get called")

        # Have to fix this part in the SSCL Model

        forgetLabelAndEntry(GatedCNN_label, GatedCNN_entry)
        forgetLabelAndEntry(SelfAttn_label, SelfAttn_entry)
        placeLabelAndEntry(SSCL_label, SSCL_entry, x, y)

    elif model == "SelfAttn":

        print("SelfAttn get called")

        forgetLabelAndEntry(GatedCNN_label, GatedCNN_entry)
        forgetLabelAndEntry(SSCL_label, SSCL_entry)
        placeLabelAndEntry(SelfAttn_label, SelfAttn_entry, x, y)

    else:
        raise ValueError


window.mainloop()
