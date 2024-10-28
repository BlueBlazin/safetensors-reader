use bytemuck;
use byteorder::{LittleEndian, ReadBytesExt};
use half::{bf16, f16};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

#[derive(Deserialize, Debug)]
struct MetadataValue {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: Vec<usize>,
}

#[derive(Deserialize, Debug)]
struct Metadata {
    #[serde(rename = "__metadata__")]
    metadata: serde_json::Value,
    #[serde(flatten)]
    items: HashMap<String, MetadataValue>,
}

#[derive(Debug)]
pub enum Tensor {
    U8 { data: Vec<u8>, shape: Vec<usize> },
    F16 { data: Vec<f16>, shape: Vec<usize> },
    Bf16 { data: Vec<bf16>, shape: Vec<usize> },
    F32 { data: Vec<f32>, shape: Vec<usize> },
}

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::U8 { shape, .. } => shape,
            Self::F16 { shape, .. } => shape,
            Self::Bf16 { shape, .. } => shape,
            Self::F32 { shape, .. } => shape,
        }
    }
}

pub struct Reader {
    pub metadata: serde_json::Value,
    pub tensors: HashMap<String, Tensor>,
}

impl Reader {
    pub fn from_file(path: &'static str) -> Result<Self, Box<dyn Error>> {
        let mut file = File::open(path)?;

        // Read the first 8 bytes to get N.
        let n_bytes = read_bytes(&mut file, 8)?;
        let n = u64::from_le_bytes(n_bytes.try_into().unwrap());

        // Read the next N bytes (the JSON data).
        let json_bytes = read_bytes(&mut file, n as usize)?;
        // Read JSON data.
        let json_data: Metadata = serde_json::from_slice(&json_bytes).expect("Invalid JSON");

        let mut ordered_keys: Vec<_> = json_data.items.keys().collect();
        ordered_keys.sort_by_key(|&key| json_data.items[key].data_offsets[0]);

        let tensors: HashMap<String, Tensor> = ordered_keys
            .into_par_iter()
            .map(|key| {
                let mut f = File::open(path).unwrap();
                let value = &json_data.items[key];
                let (start, end) = (value.data_offsets[0], value.data_offsets[1]);

                match value.dtype.as_str() {
                    "U8" => {
                        let size = end - start + 1;
                        let data = read_bytes(&mut f, size).unwrap();

                        (
                            key.to_string(),
                            Tensor::U8 {
                                data,
                                shape: value.shape.clone(),
                            },
                        )
                    }
                    "F16" => {
                        assert_eq!((end - start) % 2, 0, "Invalid alignment.");
                        let size = (end - start) / 2;
                        f.seek(SeekFrom::Start(start as u64)).unwrap();
                        let data: Vec<f16> =
                            bytemuck::allocation::cast_vec(read_bytes_u16(&mut f, size).unwrap());

                        (
                            key.to_string(),
                            Tensor::F16 {
                                data,
                                shape: value.shape.clone(),
                            },
                        )
                    }
                    "BF16" => {
                        assert_eq!((end - start) % 2, 0, "Invalid alignment.");
                        let size = (end - start) / 2;
                        f.seek(SeekFrom::Start(start as u64)).unwrap();
                        let data: Vec<bf16> =
                            bytemuck::allocation::cast_vec(read_bytes_u16(&mut f, size).unwrap());

                        (
                            key.to_string(),
                            Tensor::Bf16 {
                                data,
                                shape: value.shape.clone(),
                            },
                        )
                    }
                    "F32" => {
                        assert_eq!((end - start) % 2, 0, "Invalid alignment.");
                        let size = (end - start) / 4;
                        f.seek(SeekFrom::Start(start as u64)).unwrap();
                        let data = read_bytes_f32(&mut f, size).unwrap();

                        (
                            key.to_string(),
                            Tensor::F32 {
                                data,
                                shape: value.shape.clone(),
                            },
                        )
                    }
                    _ => panic!("The dtype {} is currently unsupported.", value.dtype),
                }
            })
            .collect();

        Ok(Reader {
            metadata: json_data.metadata,
            tensors,
        })
    }
}

fn read_bytes(file: &mut File, size: usize) -> io::Result<Vec<u8>> {
    let mut buffer = vec![0u8; size];
    file.read_exact(&mut buffer)?;
    Ok(buffer)
}

fn read_bytes_u16(file: &mut File, size: usize) -> io::Result<Vec<u16>> {
    let mut buffer = vec![0u16; size];
    file.read_u16_into::<LittleEndian>(&mut buffer)?;
    Ok(buffer)
}

fn read_bytes_f32(file: &mut File, size: usize) -> io::Result<Vec<f32>> {
    let mut buffer = vec![0.0; size];
    file.read_f32_into::<LittleEndian>(&mut buffer)?;
    Ok(buffer)
}
