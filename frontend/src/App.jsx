import SocialMediaAnalyzer from "./components/SocialMediaAnalyzer";

// The previous version carried a handleSubmit() that POSTed to /predict - a
// route the backend does not expose - and was never attached to anything.
function App() {
  return (
    <div
      className="min-h-screen w-full flex items-center justify-center relative bg-cover bg-center bg-no-repeat"
      style={{ backgroundImage: 'url("/hate.jpg")' }}
    >
      <div className="absolute inset-0 bg-black bg-opacity-30 z-10"></div>

      <div className="relative z-20 my-16 max-w-3xl w-full p-6 rounded-lg shadow-lg">
        <SocialMediaAnalyzer />
      </div>
    </div>
  );
}

export default App;
