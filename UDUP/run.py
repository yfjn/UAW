import subprocess

def runcmd():
    i=1
    while True:

        ret = subprocess.run(['/anaconda3/envs/OCR/python.exe','ATK/RepeatAdvPatchAttack.py'],
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        #print(ret.returncode)
        print("{}-th reboot".format(i))
        i+=1
runcmd()