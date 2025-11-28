import MapView from '@/components/MapView';

export default function Home() {
  return (
    <main className="flex-1 flex flex-col">
      {/* Map takes full viewport height minus header */}
      <div className="flex-1 relative">
        <MapView />
      </div>
    </main>
  );
}
