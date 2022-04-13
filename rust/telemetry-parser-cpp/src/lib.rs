extern crate libc;
use std::mem;

use std::ffi::CStr;
use std::os::raw::c_char;

use telemetry_parser::*;

#[repr(C)]
pub struct GyroData {
    pub samples: usize,
    pub timestamps: *mut f64,
    pub gyro: *mut f64,
}

#[no_mangle]
pub extern "C" fn tp_load_gyro(path_c: *const c_char, orient_c: *const c_char) -> GyroData {
    let path = unsafe { CStr::from_ptr(path_c).to_string_lossy().into_owned() };

    let orient = if orient_c != 0 as *const c_char {
        Some(unsafe { CStr::from_ptr(orient_c).to_string_lossy().into_owned() })
    } else {
        None
    };

    let mut stream = std::fs::File::open(&path).unwrap();
    let filesize = stream.metadata().unwrap().len() as usize;

    let input = Input::from_stream(&mut stream, filesize, &path).unwrap();

    // println!(
    //     "Detected camera: {} {}",
    //     input.camera_type(),
    //     input.camera_model().unwrap_or(&"".into())
    // );

    let imu_data = util::normalized_imu(&input, orient).unwrap();

    unsafe {
        let count : usize = imu_data.iter().filter(|v| v.gyro.is_some()).count();
        let gyro_out = GyroData {
            samples: count,
            timestamps: libc::malloc(mem::size_of::<f64>() as libc::size_t * count)
                as *mut f64,
            gyro: libc::malloc(mem::size_of::<f64>() as libc::size_t * count * 3)
                as *mut f64,
        };

        let mut i : usize = 0;
        let gyro_scale =  std::f64::consts::PI / 180.;
        for imd in imu_data.iter().filter(|v| v.gyro.is_some()) {
            *gyro_out.timestamps.offset(i as isize) = imd.timestamp_ms / 1000.;
            *gyro_out.gyro.offset((3 * i + 0) as isize) = imd.gyro.unwrap()[0] * gyro_scale;
            *gyro_out.gyro.offset((3 * i + 1) as isize) = imd.gyro.unwrap()[1] * gyro_scale;
            *gyro_out.gyro.offset((3 * i + 2) as isize) = imd.gyro.unwrap()[2] * gyro_scale;
            i += 1;
        }

        return gyro_out;
    }
}
