use anyhow::{bail, Result};
use candle_core::Device;

pub fn select_device(device_str: &str) -> Result<Device> {
    match device_str {
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Ok(Device::new_metal(0)?)
            }
            #[cfg(not(feature = "metal"))]
            bail!("Compile with --features metal to use Metal backend")
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Ok(Device::new_cuda(0)?)
            }
            #[cfg(not(feature = "cuda"))]
            bail!("Compile with --features cuda to use CUDA backend")
        }
        "auto" => {
            #[cfg(feature = "metal")]
            if candle_core::utils::metal_is_available() {
                return Ok(Device::new_metal(0)?);
            }
            #[cfg(feature = "cuda")]
            if candle_core::utils::cuda_is_available() {
                return Ok(Device::new_cuda(0)?);
            }
            Ok(Device::Cpu)
        }
        other => bail!("Unknown device: {other}. Use: auto, cpu, metal, cuda"),
    }
}
