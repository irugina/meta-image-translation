def parse(path):
    file = open(path, 'r')
    steps = []
    losses = []
    lines = []
    while True:
        line = file.readline()
        if line[0:14] == "loss at epoch " and "step" in line:
            line = line.split()
            lines.append(line)

            epoch = int(line[3][0:-1])
            step = int(line[5])
            total_steps = int(line[8][0:-1])
            try:
                loss = float(line[-1])
            except:
                loss = float(line[-1][0:-1])

            steps.append(float(epoch * total_steps + step) / total_steps)
            losses.append(loss)

        if not line:
            break
    file.close()
    return steps, losses, lines
