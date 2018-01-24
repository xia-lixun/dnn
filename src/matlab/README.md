# Mixing Scripts For Speech Enhancement Project

1. s = specification(): generate task specification. Edit this function to define hyper-parameters not related to the training stage.

2. data = index(s): generate meta info of the noise/speech datasets. Meta info includes level information and file locations. Also the meta info is shuffled to allow train-test split.

3. label = generate_wav(s, data, flag): flag = <'training/testing'> This will generate mixed wav files according to the flag set. Labeling info will also be recorded, written to json file. Wav files will be written to /root/<flag>/spectrum/s_*.mat

4. feature(s, label, flag): generate spectrum data to /root/training/spectrum/s_*.mat   