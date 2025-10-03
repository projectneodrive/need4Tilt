# RTK Vehicle Trace Dataset

This repository contains a collection of fused GPS + inertial navigation traces from a passenger vehicle.  
The data consists of ~30 hours of driving in and around Oxford (UK), sampled at **10 Hz**. Each run is stored in its own folder as a CSV file.

it comes originally from https://robotcar-dataset.robots.ox.ac.uk/ground_truth/ 

---

## Dataset structure

```
rtk/
└── rtk/
    ├── 2014-11-11-11-06-25/
    │ └── rtk.csv
    ├── 2014-11-14-16-34-33/
    │ └── rtk.csv
    ├── 2014-11-18-13-20-12/
    │ └── rtk.csv
    └── ...
```

There are **73 CSV files** corresponding to different driving sessions, recorded between 2014–2015.

---

## File format

Each `rtk.csv` file contains the fused GPS+IMU navigation solution with the following columns:

| Column            | Unit  | Description |
|-------------------|-------|-------------|
| `timestamp`       | µs since UNIX epoch | Logging time |
| `latitude`        | deg   | WGS84 latitude |
| `longitude`       | deg   | WGS84 longitude |
| `altitude`        | m     | Altitude above ellipsoid |
| `northing`        | m     | UTM northing |
| `easting`         | m     | UTM easting |
| `down`            | m     | Down position in NED frame |
| `utm_zone`        | —     | UTM zone (e.g. 30U) |
| `velocity_north`  | m/s   | North velocity |
| `velocity_east`   | m/s   | East velocity |
| `velocity_down`   | m/s   | Down velocity |
| `roll`            | rad   | Vehicle roll angle |
| `pitch`           | rad   | Vehicle pitch angle |
| `yaw`             | rad   | Vehicle yaw angle |



- Sample rate: **~10 Hz** (Δt ≈ 0.1 s)  
- Total driving time: **≈ 30 hours**

---

## License

The dataset is released under the [Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

You are free to:

- **Share** — copy and redistribute the material in any medium or format  
- **Adapt** — remix, transform, and build upon the material  

Under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.  
- **NonCommercial** — You may not use the material for commercial purposes.  
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.  

---

## Attribution

https://robotcar-dataset.robots.ox.ac.uk/ground_truth/

If you use this dataset in your publications, please cite:

> W. Maddern, G. Pascoe, C. Linegar, and P. Newman,  
> *1 Year, 1000km: The Oxford RobotCar Dataset*,  
> The International Journal of Robotics Research (IJRR), 2017.  
> [PDF](https://robotcar-dataset.robots.ox.ac.uk/images/robotcar_ijrr.pdf) | [BibTeX](https://robotcar-dataset.robots.ox.ac.uk/publications/robotcar_ijrr.bib)

If you use the released **RTK ground truth**, please also cite:

> W. Maddern, G. Pascoe, M. Gadd, D. Barnes, B. Yeomans, and P. Newman,  
> *Real-time Kinematic Ground Truth for the Oxford RobotCar Dataset*,  
> arXiv preprint arXiv:2002.10152, 2020.  
> [PDF](https://arxiv.org/pdf/2002.10152.pdf) | [BibTeX](https://arxiv.org/bibtex/2002.10152)


---

## Notes

- Results, plots, code, and analyses derived from this dataset (e.g., suspension models, travel time estimates) are not subject to the dataset license — only the dataset files themselves are.  
