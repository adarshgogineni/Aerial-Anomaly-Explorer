export default function Home() {
  return (
    <main className="min-h-screen p-8">
      <div className="container mx-auto">
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-8 shadow-lg">
          <h2 className="text-3xl font-bold mb-4">Welcome to UAP Explorer</h2>
          <p className="text-lg mb-4">
            An interactive web application that visualizes UFO/UAP sighting reports
            on an interactive map and uses machine learning to highlight spatiotemporal
            anomalies and narrative patterns in the data.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
            <div className="bg-white dark:bg-gray-700 p-6 rounded-lg">
              <h3 className="text-xl font-semibold mb-2">Spatiotemporal Analysis</h3>
              <p>Explore when and where UAP reports cluster or spike over time.</p>
            </div>
            <div className="bg-white dark:bg-gray-700 p-6 rounded-lg">
              <h3 className="text-xl font-semibold mb-2">Anomaly Detection</h3>
              <p>ML-powered scoring identifies statistically unusual reports and regions.</p>
            </div>
            <div className="bg-white dark:bg-gray-700 p-6 rounded-lg">
              <h3 className="text-xl font-semibold mb-2">Pattern Discovery</h3>
              <p>Clustered descriptions reveal different types of reported phenomena.</p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
