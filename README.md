# MEMENTO: Memory-Guided Memetic Code-as-Policy Evolution

MEMENTO is a framework for evolving executable code-as-policy programs for robotic manipulation and embodied-control tasks. The repository contains the evolutionary search code, prompts, selected policies, teaser GIFs, and full demonstration videos used to illustrate the learned behaviours.

## Demos

The GIFs below are stored in `assets/` and play directly on the GitHub page.

<table>
  <tr>
    <td align="center" width="50%">
      <img src="./assets/physical_franka_teaser.gif" width="420"><br>
      <b>Physical Franka</b>
    </td>
    <td align="center" width="50%">
      <img src="./assets/robosuite_teaser.gif" width="420"><br>
      <b>Robosuite Simulation</b>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="./assets/thor_demo1.gif" width="420"><br>
      <b>AI2-THOR Demo 1</b>
    </td>
    <td align="center" width="50%">
      <img src="./assets/thor_demo2.gif" width="420"><br>
      <b>AI2-THOR Demo 2</b>
    </td>
  </tr>
</table>

Full-resolution videos are available in the `videos/` folder.

## Installation

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Repository Structure

```text
MEMENTO-Memory-Guided-Memetic-Code-as-Policy-Evolution/
├── MEMENTO/             # Core source code, prompts, environment, and utilities
├── best_policies/       # Selected best-performing evolved policy programs
├── assets/              # Teaser GIFs shown in this README
├── videos/              # Full demonstration videos
├── requirements.txt     # Python dependencies
└── README.md
```

## Main Components

* `MEMENTO/main.py`: main entry point for running the evolutionary pipeline.
* `MEMENTO/utils.py`: evaluation, execution, and helper utilities.
* `MEMENTO/inference.py`: model-query and inference utilities.
* `MEMENTO/env_tower_heavy_dr.py`: Robosuite Franka environment used for manipulation experiments.
* `MEMENTO/prompts/`: prompt templates for code-as-policy generation and refinement.
* `MEMENTO/cfg/`: configuration files.
* `best_policies/`: selected evolved policies for inspection and reuse.
* `assets/`: compressed GIF teasers for GitHub rendering.
* `videos/`: full demo videos.

## Assets

The README expects the following files:

```text
assets/
├── physical_franka_teaser.gif
├── robosuite_teaser.gif
├── thor_demo1.gif
└── thor_demo2.gif
```

## Best Policies

The best-performing evolved code-as-policy programs are stored in:

```text
best_policies/
```

These files provide compact examples of the final executable policies produced by the MEMENTO search process.

## Videos

Full demonstration videos are stored in:

```text
videos/
```
