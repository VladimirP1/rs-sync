### About
This project was done as a part of a bachelor's thesis. Its goal was to improve synchronisation of rolling-shutter videos in GyroFlow.
The thesis full text (in Russian) is [here](thesis-text.pdf).

This repository contains the C++ version which was used for development, then the code has been translated to Rust by AdrianEddy.
The Rust version is [here](https://github.com/gyroflow/rs-sync).

### Build steps
1. `cargo build --release` to build the telemetry-parser wrapper
2. Clone vcpkg into `ext/vcpkg`, bootstrap it
3. Do a normal CMake build

### Running
The executable `core_testcode` accepts a single argument on the command line which is the path to a JSON file like this:
```json
    {
        "input": {
            "video_path": "GX012440.MP4",
            "gyro_path": "GX012440.MP4",
            "gyro_orientation": "yZX",
            "frame_range": [
                3900,
                7200
            ],
            "lens_profile": {
                "type": "file",
                "path": "lens.txt",
                "name": "hero6_27k_43"
            },
            "initial_guess": 0,
            "use_simple_presync": true,
            "simple_presync_radius": 200,
            "simple_presync_step": 2
        },
        "params": {
            "sync_window": 60,
            "syncpoints_format": "auto",
            "syncpoint_distance" : 120
        },
        "output": {
            "csv_path": "sync_GX012440.csv"
        }
    }

```

***WARNING:*** `gyro_orientation` cannot be blindly copied from GyroFlow, because it is not defined in the same way.
I used the code in `src/guess_orient_json.cpp` on the `iter2` branch for finding the orientation. It basically tries all orientations and selects the one with lowest loss function value.

***NOTE:*** Time in JSON is specified in miliseconds.

The file mentioned in `input.lens_profile.path` file should contain the lens profile definitions in the following form:
```
<camera name> <readout time in seconds> <fx> <fy> <cx> <cy> <k1> <k2> <k3> <k4>
```

For example for a GoPro Hero 6 in 2.7k 4:3 mode this would be:
```
hero6_27k_43 0.01111 1186 1186 1355.389 1020.317 0.04440465777694087 0.01946789951179939 -0.004476697539343917 -0.002042912877740792
```

### Using as a library
There's an interface called `ISyncProblem`, instances can be created using `CreateSyncProblem()`.

1. Gyro quaternions have to be provided using one of `SetGyroQuaternions` overloads. One overload is for fixed sample rate data, the other is for variable. Variable sample rate data will be interpolated into fixed sample rate data internally.
2. Tracking data has to be provided for each frame using `SetTrackResult`. If there's no data for some frames, just skip them. This consists of rays (as unit vectors) corresponding to tracked image features from current to next frame and their timestamps.
3. Call `PreSync` to find coarse sync. This is basically a brute-force search on an approxximation of the loss function. The loss function can be exported for plotting using the `DebugPreSync` method.
4. Use the offset found with `PreSync` as an initial delay for `Sync`. The example code does this multiple times. This seemed to work better because there's every `Sync` the translation directions for each frame and some optimization hyperparameters are reestimated.

***NOTE:*** Time in the library interface is in seconds unless otherwise stated using the `_us` suffix.