# Welcome to Dough v. 0.9.8 (beta)

**⬇️ Scroll down for Setup Instructions - Currently available on Linux & Windows, hosted version coming soon.**

Dough is a tool for crafting videos with AI. Our goal is to give you enough control over video generations that you can make beautiful creations of anything you imagine that feel uniquely your own.

To achieve this, we allow you to guide video generations with precision using a combination of images (via [Steerable Motion](https://github.com/banodoco/steerable-motion)) examples videos (via [Motion Director](https://github.com/ExponentialML/AnimateDiff-MotionDirector)).

Below is brief overview and some examples of outputs:

### With Dough, you can makes guidance frames using Stable Diffusion XL, IP-Adapter, Fooocus Inpainting, and more:

<img src="https://github.com/banodoco/Dough/assets/34690994/698d63f5-765c-4cf2-94f4-7943d241a6ea" width="800">

### You can then assemble these frames into shots that you can granularly edit:

<img src="https://github.com/banodoco/Dough/assets/34690994/1080ed90-b829-47cd-b946-de49a7a03b2a" width="800">

### And then animate these shots by defining parameters for each frame and selecting guidance videos via Motion LoRAs:

<img src="https://github.com/banodoco/Dough/assets/34690994/95ec3ec3-5143-40e9-88ba-941ce7e2dec9" width="800">

### As an example, here's a video that's guided with just images on high strength:

<img src="https://github.com/banodoco/Dough/assets/34690994/cc88ca21-870d-4b96-b9cc-39698fc5fd2f" width="800">

### While here's a more complex one, with low strength images driving it alongside a guidance video:

<img src="https://github.com/banodoco/Dough/assets/34690994/5c2edc07-8aa3-402f-b119-345db26df8b9" width="800">

### And here's a more complex example combining high strength guidance with a guidance video strongly influencing the motion:

<img src="https://github.com/banodoco/Dough/assets/34690994/e5d70cc3-03e2-450d-8bc7-b6d1a920af4a" width="800">



### We're obviously very biased think that it'll be possible to create extraordinarily beautiful things with this and we're excited to see what you make! Please share stuff you made in our [Discord](https://discord.com/invite/8Wx9dFu5tP).


# Setup Instructions

<details>
  <summary><b>Setting up on Runpod (click to expand)</b></summary>

  
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


## Instructions for Linux:

### Install the app

This commands sets up the app. Run this only the first time, after that you can simply start the app using the next command.
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

## Instructions for Windows:

### Open Powershell in Administrator mode

Open the Start menu, type Windows PowerShell, right-click on Windows PowerShell, and then select Run as administrator.

### Install the app

Install MS C++ Redistributable (if not already present) - https://aka.ms/vs/16/release/vc_redist.x64.exe

### Navigate to Documents

Make sure you're in the documents folder by running the following command:

```bash
cd ~\Documents
```

### Run the setup script

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

---

### Using personal ComfyUI instance
You can provide the path to your ComfyUI instance in the "App Settings" section of Dough


If you're having any issues, please share them in our [Discord](https://discord.com/invite/8Wx9dFu5tP).

# Troubleshooting

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
