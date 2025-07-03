import React, { useState } from 'react';

function App() {
  const [url, setUrl] = useState('');
  const [results, setResults] = useState([]);

  const handleSubmit = async () => {
    try {
      const res = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_url: url }),
      });

      if (!res.ok) {
        throw new Error(`Server Error: ${res.status}`);
      }

      const data = await res.json();
      setResults(data);
    } catch (err) {
      console.error('❌ 요청 실패:', err);
      setResults([]);
    }
  };

  return (
    <div className="p-4 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">🍔 Image Similarity Search</h1>

      <input
        value={url}
        onChange={e => setUrl(e.target.value)}
        className="border p-2 w-full mb-2"
        placeholder="Enter Image URL"
      />

      <button
        onClick={handleSubmit}
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
      >
        Search
      </button>

      <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
        {results.map((item, i) => (
          <div key={i} className="text-center">
            <img
              src={`http://localhost:8000/images/${item.uri}`}
              alt={item.name}
              className="w-full h-32 object-cover rounded"
            />
            <div className="mt-1 text-sm">
              {item.name} ({item.distance.toFixed(2)})
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
