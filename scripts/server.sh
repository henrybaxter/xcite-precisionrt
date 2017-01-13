sudo apt update && sudo apt dist-upgrade -y -qq
sudo apt install build-essential libbz2-dev libssl-dev libreadline-dev libgeos-dev awscli \
	libsqlite3-dev libfreetype6-dev pkg-config texlive-full gfortran sysstat grace openscad -y -qq

# sudo apt install python3-pip -y -qq
curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
pyenv install 3.6.0
pyenv virtualenv 3.6.0 xcite
pyenv global xcite

#sudo add-apt-repository ppa:jonathonf/python-3.6
#sudo apt-get update
#sudo apt-get install python3.6
# sudo hostname egsX
# sudo vim /etc/hosts

# my ssh key

git clone git@github.com:henrybaxter/beamdpr.git
git clone git@github.com:henrybaxter/EGSnrc.git
git clone git@github.com:henrybaxter/xcite-precisionrt.git

git config --global push.default simple
git config --global user.name "Henry Baxter"
git config --global user.email henry.baxter@gmail.com

~/.vimrc
set bg=dark
set et
set sts=4
set ts=4
set sw=4

curl https://sh.rustup.rs -sSf | sh
source ~/.cargo/env
cd beamdpr
cargo install
cd ..

cd EGSnrc

# into ~/.profile
export EGS_HOME=/home/ubuntu/EGSnrc/egs_home/
export EGS_CONFIG=/home/ubuntu/EGSnrc/HEN_HOUSE/specs/linux.conf
source /home/ubuntu/EGSnrc/HEN_HOUSE/scripts/egsnrc_bashrc_additions
cd ..

cd xcite-precisionrt
pip install --upgrade -r requirements.txt
curl -L https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp35-cp35m-linux_x86_64.whl -o tensorflow-0.12.1-cp36-cp36m-linux_x86_64.whl
pip install --upgrade tensorflow-0.12.1-cp36-cp36m-linux_x86_64.whl


# from local
scp allkV.pegs4dat egs2:EGSnrc/HEN_HOUSE/pegs4/data/






