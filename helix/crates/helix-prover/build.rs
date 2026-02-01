//! Build script for helix-prover.
//!
//! Handles CUDA kernel compilation when the cuda feature is enabled.

fn main() {
    // Only compile CUDA kernels if the cuda feature is enabled
    #[cfg(feature = "cuda")]
    compile_cuda_kernels();

    // Rerun if CUDA source files change
    println!("cargo:rerun-if-changed=src/cuda/kernels/");
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernels() {
    use std::env;
    use std::path::PathBuf;
    use std::process::Command;

    // Check if nvcc is available
    let nvcc_result = Command::new("nvcc").arg("--version").output();
    if nvcc_result.is_err() {
        println!("cargo:warning=NVCC not found, skipping CUDA kernel compilation");
        println!("cargo:warning=To enable CUDA support, install CUDA Toolkit");
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_src_dir = PathBuf::from("src/cuda/kernels");

    // Check if CUDA source files exist
    if !cuda_src_dir.exists() {
        println!("cargo:warning=CUDA kernel directory not found, skipping compilation");
        return;
    }

    // Compile CUDA kernels
    let cuda_files = [
        "field_ops.cu",
        "msm.cu",
        "ntt.cu",
        "runtime.cu",
    ];

    let mut objects = Vec::new();

    for cuda_file in &cuda_files {
        let src_path = cuda_src_dir.join(cuda_file);
        if !src_path.exists() {
            println!("cargo:warning=CUDA file {} not found, skipping", cuda_file);
            continue;
        }

        let obj_name = cuda_file.replace(".cu", ".o");
        let obj_path = out_dir.join(&obj_name);

        // Compile .cu to .o
        let status = Command::new("nvcc")
            .args([
                "-c",
                "-O3",
                "--generate-code=arch=compute_70,code=sm_70", // Volta
                "--generate-code=arch=compute_75,code=sm_75", // Turing
                "--generate-code=arch=compute_80,code=sm_80", // Ampere
                "--generate-code=arch=compute_86,code=sm_86", // Ampere GA102
                "--generate-code=arch=compute_89,code=sm_89", // Ada Lovelace
                "-Xcompiler", "-fPIC",
                "-o",
            ])
            .arg(&obj_path)
            .arg(&src_path)
            .status();

        match status {
            Ok(s) if s.success() => {
                objects.push(obj_path);
            }
            Ok(s) => {
                println!("cargo:warning=NVCC failed with status {} for {}", s, cuda_file);
            }
            Err(e) => {
                println!("cargo:warning=Failed to run NVCC for {}: {}", cuda_file, e);
            }
        }
    }

    if objects.is_empty() {
        println!("cargo:warning=No CUDA objects compiled");
        return;
    }

    // Create static library from object files
    let lib_path = out_dir.join("libhelix_cuda.a");
    let mut ar_args = vec!["rcs".to_string(), lib_path.to_str().unwrap().to_string()];
    ar_args.extend(objects.iter().map(|p| p.to_str().unwrap().to_string()));

    let ar_status = Command::new("ar")
        .args(&ar_args)
        .status();

    match ar_status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-link-search=native={}", out_dir.display());
            println!("cargo:rustc-link-lib=static=helix_cuda");

            // Link CUDA runtime libraries
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cuda");

            // On Linux, also link stdc++
            #[cfg(target_os = "linux")]
            println!("cargo:rustc-link-lib=stdc++");
        }
        Ok(s) => {
            println!("cargo:warning=ar failed with status {}", s);
        }
        Err(e) => {
            println!("cargo:warning=Failed to run ar: {}", e);
        }
    }
}
