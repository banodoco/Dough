# Dough is an open-source art tool for steering video with precision

**⬇️ Scroll down for Setup Instructions - Currently available on Linux & Windows computers with more than 12GB of VRAM, hosted version coming soon.**

Dough is a tool for crafting videos with AI. Our goal is to give you enough control over video generations that you can make beautiful creations of anything you imagine that feel uniquely your own.

To achieve this, we allow you to guide video generations with precision using a combination of images (via [Steerable Motion](https://github.com/banodoco/steerable-motion)) and examples videos (via [Motion Director](https://github.com/ExponentialML/AnimateDiff-MotionDirector)) with granular motion settings for each frame.

To start, with Dough, you generate hundreds of images in the right style and theme using the Inspiration Engine:

![images (2)](https://github.com/user-attachments/assets/ebdb923e-7d4e-4dee-8cce-db62d5c62be2)
<br>
You can then assemble these frames into shots that you can granularly edit:
<br>

![edit (1)](https://github.com/user-attachments/assets/405af368-b2ad-406a-93aa-942d84dd48cc)

And then animate these with granular settings for each frame and Motion LoRAs:
<br>

![animate (1)](https://github.com/user-attachments/assets/2ced74e1-88c7-4c49-a3f4-942bb00f1893)

As an example of it in action, here's 4 images steering, alongside a LoRA trained on a video:
<br>
<br>
<img src="https://github.com/banodoco/Dough/assets/34690994/e5d70cc3-03e2-450d-8bc7-b6d1a920af4a" width="800">

While here are four abstract images animated by each of the different workflows available inside Dough:

![basic workflows](https://github.com/banodoco/steerable-motion/blob/main/demo/basic_workflows.gif)

We're obviously very biased think that it'll be possible to create extraordinarily beautiful things with this and we're excited to see what you make! Please share stuff you made in our [Discord](https://discord.com/invite/8Wx9dFu5tP).

## Artworks created with Dough

You can see a selection of artworks created with Dough [here](https://banodoco.ai/Dough#some-weird-beautiful-and-interesting-things-people-have-made-with-dough-and-steerable-motion-the-technology-behind-it).

## Setup Instructions

### Recommended Setup for Linux & Windows - Pinokio:

Pinokio is a web browser for launching and managing AI apps. We recommend you use it for installing Dough.

Instructions:

1) Download [Pinokio Browser](https://pinokio.computer/)

2) Click into Discover, search "Dough" select the option by "Banodoco" and press the download button

3) Once it's downloaded, press the Install button. This will take a few minutes

4) Once it's installed, press the Run button and it should launch

### Other Setup Instructions:

<details>
  <summary><b>Setting up on Runpod</b></summary>

  
1) We recommend setting up persistent storage for a quick setup and for your projects to persist. To get it going, click into “Storage”, select “New Network Volume”. 50GB should be more than enough to start.


2) Select a machine - any should work, but we recommend a 4090.


3) During setup, open the relevant ports for Dough like below:


<img src="https://github.com/banodoco/Dough/assets/34690994/102bc6fe-0962-493f-b11a-9dfa22501bdd" width="600">

<img src="https://github.com/banodoco/Dough/assets/34690994/1b9ff4d7-960e-496c-83ae-306c0dfa623d" width="600">


4) When you’ve launched the pod, click into Jupyter Notebook:

<img src="https://github.com/banodoco/Dough/assets/34690994/9a0b6b54-ae53-4571-8131-165c4bacc909" width="600">

<img src="https://github.com/banodoco/Dough/assets/34690994/86b31523-7457-43b2-ad68-99e62689c32f" width="600">


5) Follow the “Setup for Linux” below and come back here when you’ve gone through them.


6) Once you’re done that, grab the IP Address for your instance:

<img src="https://github.com/banodoco/Dough/assets/34690994/35aed283-fa47-494e-924e-0263b84be2b2" width="600">

<img src="https://github.com/banodoco/Dough/assets/34690994/2bdb9363-9138-49bd-a2b9-69961e744f7a" width="600">

<img src="https://github.com/banodoco/Dough/assets/34690994/a2a83ee6-149e-44aa-b00a-d36e42320bb4" width="600">

Then form put these into this form with a : between them like this:

{Public ID}:{External Pair Value}

In the above example, that would make it:

213.173.108.4:14810

Then go to this URL, and it should be running!

**Important:** remember to terminate the instance once you’re done - you can restart it by following the instructions from step 3 above.

</details>


<details>
  <summary><b>Manual installation instructions for Linux</b></summary>

### Install the app

This commands sets up the app. Run this only the first time, after that you can simply start the app using the next command.

Local GPU mode
```bash
curl -sSL https://raw.githubusercontent.com/banodoco/Dough/green-head/scripts/linux_setup.sh | bash
```

### Enter the folder

In terminal, run:
```bash
cd Dough
```

### Run the app

you can run the app using 

```bash
source ./dough-env/bin/activate && ./scripts/entrypoint.sh
```
</details>


<details>
  <summary><b>Manual installation instructions for Windows</b></summary>

### Open Powershell in Administrator mode

Open the Start menu, type Windows PowerShell, right-click on Windows PowerShell, and then select Run as administrator.
Then run this command ```Set-ExecutionPolicy RemoteSigned```

**NOTE:** Make sure you have Python3.10 installed and set as the default Python version

### Install the app

Install MS C++ Redistributable (if not already present) - https://aka.ms/vs/16/release/vc_redist.x64.exe

### Navigate to Documents

Make sure you're in the documents folder by running the following command:

```bash
cd ~\Documents
```

### Run the setup script

Local GPU mode
```bash
iwr -useb "https://raw.githubusercontent.com/banodoco/Dough/green-head/scripts/windows_setup.bat" -OutFile "script.bat"
Start-Process "cmd.exe" -ArgumentList "/c script.bat"
```

### Enter the folder
In Powershell, run:
```bash
cd Dough
```

### Run the app

```bash
. .\dough-env\Scripts\activate ; .\scripts\entrypoint.bat
```
</details>


## Troubleshooting


<details>
  <summary><b>Common problems (click to expand)</b></summary>

<details>
  <summary><b>Issue during installation</b></summary>
  
- Make sure you are using python3.10
- If you are on Windows, make sure permissions of the Dough folder are not restricted (try to grant full access to everyone)
- Double-check that you are not inside any system-protected folders like system32
- Install the app in admin mode. Open the powershell in the admin mode and run "Set-ExecutionPolicy RemoteSigned". Then follow the installation instructions given in the readme
- If all of the above fail, try to run the following instructions one by one and report which one is throwing the error
  ```bash
  call dough-env\Scripts\activate.bat
  python.exe -m pip install --upgrade pip
  pip install -r requirements.txt
  pip install websocket
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install -r comfy_runner\requirements.txt
  pip install -r ComfyUI\requirements.txt
  ```
</details>
<details>
  <summary><b>Unable to locate credentials</b></summary>
  Make a copy of ".env.sample" and rename it to ".env"
</details>
<details>
  <summary><b>Issue during runtime</b></summary>

- If a particular node inside Comfy is throwing an error then delete that node and restart the app
- Make sure you are using python3.10 and the virtual environment is activated
- Try doing "git pull origin main" to get the latest code
</details>
<details>
  <summary><b>Generations are in progress for a long time</b></summary>

- Check the terminal if any progress is being made (they can be very slow, especially in the case of upscaling)
- Cancel the generations directly from the sidebar if they are stuck
- If you don't see any logs in the terminal, make sure no other program is running at port 12345 on your machine as Dough uses that port
</details>
<details>
  <summary><b>Some other error?</b></summary>
  
  Drop in our [Discord](https://discord.com/invite/8Wx9dFu5tP).
</details>
</details>

## Interested in joining a community of people who are pushing open AI art models to their technical and artistic limits?

Drop in our [Discord](https://discord.com/invite/8Wx9dFu5tP).

