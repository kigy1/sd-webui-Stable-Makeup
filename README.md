# sd-webui-Stable-Makeup

![sm_teaser](https://github.com/kigy1/sd-webui-Stable-Makeup/assets/136764953/4c74892c-f52f-4db8-8da3-8be9e32a7478)

## About

The extension `sd-webui-Stable-Makeup` brings Xiaojiu-z's Stable-Makeup to the stable-diffusion-webui, making it easy to use. You can find the original project here: [Xiaojiu-z/Stable-Makeup](https://github.com/Xiaojiu-z/Stable-Makeup).

I'm new to making extensions, so if there are any problems, feel free to help me fix them. I've tried to make it user-friendly by allowing it to download all models by itself.

This extension has advantages over the original Stable-Makeup, such as working with smaller faces. However, it also has some disadvantages, like making the process slower. You can downscale the image to something like 512x512 to make it much faster.

![webui-vs-orginal](https://github.com/kigy1/sd-webui-Stable-Makeup/assets/136764953/ab618e8f-79d4-49d6-a1e0-873c4a4d7e07)

![Screenshot 2024-05-23 005128](https://github.com/kigy1/sd-webui-Stable-Makeup/assets/136764953/a229d5f2-b23f-48eb-b7b8-0ae5d029aba7)

## Installation

1. Open the Extensions tab.
2. Click on Install from URL.
3. Enter the URL for the extension's git repository.
4. Click Install.
5. Restart WebUI.
6. sometimes at first you need to run automatic1111 two times or more to install and download the necessary

### Important Notes:

1. The first installation will take some time.
2. Also, add `--disable-safe-unpickle` to `webui-user.bat` next to `COMMANDLINE_ARGS` (I wish if someone can help me fix this).
3. Don't use `--no-half`, as this will make it slower.
4. sometimes at first you need to run automatic1111 two times or more to install and download the necessary

## To-Do List

- [ ] Make it work with many faces
- [ ] Make it faster by adding an option to downscale images
- [ ] Make it work with Stable Diffusion WebUI model (need someone to help with that)
- [ ] Try to remove box from some photos by adding seg (for now, if you face this issue, upscale the image)
