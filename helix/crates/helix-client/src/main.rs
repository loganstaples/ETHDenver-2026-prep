use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "helix")]
#[command(about = "Helix CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Init,
    Join,
    Status,
    Query,
    Export,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Init => println!("Initializing..."),
        Commands::Join => println!("Joining..."),
        Commands::Status => println!("Checking status..."),
        Commands::Query => println!("Querying..."),
        Commands::Export => println!("Exporting..."),
    }
}
