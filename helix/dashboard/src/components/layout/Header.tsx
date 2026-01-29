export default function Header() {
    return (
        <header className="bg-gray-800 shadow p-4">
            <div className="flex justify-between items-center">
                <h2 className="text-xl font-semibold">Helix</h2>
                <button className="bg-blue-600 px-4 py-2 rounded">Connect Wallet</button>
            </div>
        </header>
    );
}
