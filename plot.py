import matplotlib.pyplot as plt
import re
import sys
def main(log_name):
    with open(log_name, 'r') as f:
        lines = f.read().splitlines()

    epoch = int(re.search(r'\d+',lines[2]).group())
    train_acc = []; train_micro = []; train_macro = []
    val_acc = []; val_micro = []; val_macro = []
    train_loss = []; val_loss = []

    for ii in range(epoch):

        acc = re.search(r'(?<=[\)][\:][\s])(\d|\.)+',lines[ii*7+5]).group()
        train_acc.append(float(acc))

        acc = re.search(r'(?<=[\)][\:][\s])(\d|\.)+',lines[ii*7+7]).group()
        val_acc.append(float(acc))

        micro = re.search(r'(?<=[M][i][c][r][o][\-][F][1][\:][\s])(\d|\.)+',lines[ii*7+6]).group()
        train_micro.append(float(micro))
        macro = re.search(r'(?<=[M][a][c][r][o][\-][F][1][\:][\s])(\d|\.)+',lines[ii*7+6]).group()
        train_macro.append(float(macro))

        micro = re.search(r'(?<=[M][i][c][r][o][\-][F][1][\:][\s])(\d|\.)+',lines[ii*7+8]).group()
        val_micro.append(float(micro))
        macro = re.search(r'(?<=[M][a][c][r][o][\-][F][1][\:][\s])(\d|\.)+',lines[ii*7+8]).group()
        val_macro.append(float(macro))

        l = re.search(r'(?<=[s][\:][\s])(\d|\.)+',lines[ii*7+5]).group()
        train_loss.append(float(l))
        l = re.search(r'(?<=[s][\:][\s])(\d|\.)+',lines[ii*7+7]).group()
        val_loss.append(float(l))


    plt.ion()
    xaxis = range(epoch)
    plt.figure(1)
    plt.plot(xaxis, train_acc, marker='o', mec='r', mfc='w',label='Train Accuracy')
    plt.plot(xaxis, val_acc, marker='^', ms=10,label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Food Categorization Accuracy')
    plt.legend()

    plt.figure(2)
    plt.plot(xaxis, train_micro, marker='o', mec='r', mfc='w',label='Train Micro-F1')
    plt.plot(xaxis, val_micro, marker='^', ms=10,label='Validation Micro-F1')
    plt.xlabel('Epoch')
    plt.ylabel('Micro-F1')
    plt.title('Ingredients Recognition Micro-F1 Value')
    plt.legend()

    plt.figure(3)
    plt.plot(xaxis, train_macro, marker='o', mec='r', mfc='w',label='Train Macro-F1')
    plt.plot(xaxis, val_macro, marker='^', ms=10,label='Validation Macro-F1')
    plt.xlabel('Epoch')
    plt.ylabel('Macro-F1')
    plt.title('Ingredients Recognition Macro-F1 Value')
    plt.legend()

    plt.figure(4)
    plt.plot(xaxis, train_loss, marker='o', mec='r', mfc='w',label='Train Loss Function Value')
    plt.plot(xaxis, val_loss, marker='^', ms=10,label='Validation Loss Function Value')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Loss Function Value')
    plt.legend()

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    args = sys.argv

    main(args[1])

    quit(0)
