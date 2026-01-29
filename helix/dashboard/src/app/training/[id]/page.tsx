export default function TrainingRunPage({ params }: { params: { id: string } }) {
    return (
        <div className="p-6">
            <h1 className="text-2xl font-bold mb-4">Training Run: {params.id}</h1>
        </div>
    );
}
