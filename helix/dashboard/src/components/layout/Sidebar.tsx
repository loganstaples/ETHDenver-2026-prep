export default function Sidebar() {
    return (
        <aside className="w-64 bg-gray-800 flex flex-col p-4">
            <nav className="space-y-2">
                <a href="/" className="block p-2 hover:bg-gray-700 rounded">Dashboard</a>
                <a href="/training" className="block p-2 hover:bg-gray-700 rounded">Training</a>
                <a href="/model" className="block p-2 hover:bg-gray-700 rounded">Models</a>
                <a href="/proofs" className="block p-2 hover:bg-gray-700 rounded">Proofs</a>
                <a href="/nodes" className="block p-2 hover:bg-gray-700 rounded">Nodes</a>
            </nav>
        </aside>
    );
}
