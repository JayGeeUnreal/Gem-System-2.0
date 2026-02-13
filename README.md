# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Installation <br>
Now, for my setup i´m using <br>
Anaconda3_Py3_13-2025.06-0-Windows-x86_64.exe<br>
Voicemeter - https://vb-audio.com/Voicemeeter/banana.htm<br>
*Voicemeter Banana is not needed for basic usage, but needed for all functions to work.<br>

Ports used :<br>
MCP&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5000<br>
Chat Interceptor	&nbsp;&nbsp;&nbsp;&nbsp;8889<br>
StyleTTS2		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;13300/tts<br>
LuxTTS		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;13300/tts<br>
Server ??		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5050<br>
OSC(In Unreal)	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10000/chat/message<br>
Vision Service		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5001/sca<br>
			&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5001/get_image<br>
NeurosyncLocalAPI	9000<br>
	End point	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5000<br>
Watcher/Neurosync	11111(Send to Unreal Engine)<br>
Listen (Ears)		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5000 (sends to)<br>
Chat Interceptor	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;14300 (ssn_chat_saver.py)<br>
			&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ssn_chat_saver.py:360:@app.route('/api/live', methods=['GET'])<br>
			&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ssn_chat_saver_inject.py:518:@app.route('/api/live', methods=['GET'])<br>
Install Social Stream Ninja<br>
Cuda Toolkit - https://developer.nvidia.com/cuda-downloads<br>
________________________________________________________________________________
Make a folder for example 
C:\Users\jorge\Documents\AI\Gem_2_0

Go into the folder and run
git clone https://github.com/JayGeeUnreal/Gem-AI-System-2.0.git

Go into Gem-AI-System-2.0

Start an Anaconda Prompt

Run (example)
conda create --name mcp_env_1 python=3.10 -y
conda activate  mcp_env_1
Cd into (Example)
C:\Users\jorge\Documents\AI\Gem_2_0\Gem-AI-System-2.0

Use nvcc –version to get the cuda version you have installed.
# Then go to https://pytorch.org/get-started/locally/ to get the command to run.
For me I run
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130 

Run
pip install requirements.txt
or
conda env create -f mcp_env_1.yml


Neurosync
Download the model from Hugging Face and put it in
Gem-AI-System-2.0\Neurosync\NeuroSync_Local_API\utils\model

StyleTTS2:
Download and install eSpeak NG
https://github.om/espeak-ng/releases
Set enviroment variables
PHONEMIZER_ESPEAK_PATH = C:\Program Files\eSpeak NG
PHONEMIZER_ESPEAK_LIBRARY =  C:\Program Files\eSpeak NG\libespeak-ng.dll
Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD

Download Gradio demo files to YOUR_PATH\Gem-AI-System-2.0\StyleTTS2 (Not needed..)
https://huggingface.co/spaces/styletts2/styletts2/tree/main
ljspeechimportable.py
styletts2importable.py
app.py

LuxTTS
To get GPU acceleration you need to run the following
Make sure PyTorch version matches the CUDA version installed on your system
    1. conda install cudatoolkit cudnn -c conda-forge -y
    2. conda install -c pytorch pytorch
    3. pip install git+https://github.com/k2-fsa/k2



To Come :

Need to update the Gemini method since they changed some things... There is a warning but it can be ignored for now.....

Need to bug test more

Need to write new documentation
