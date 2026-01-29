use std::io::Result;

fn main() -> Result<()> {
    prost_build::compile_protos(
        &[
            "helix/common.proto",
            "helix/training.proto",
            "helix/gradient.proto",
            "helix/proof.proto",
            "helix/network.proto",
        ],
        &["."],
    )?;
    Ok(())
}
