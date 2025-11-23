import PoseCamera from "./components/PoseCamera";
import "./App.css";

function App() {
  return (
    <div className="app">
      <div className="appShell">
        <header className="appHeader">
          <div className="eyebrow">Push-up session</div>
          <div className="titleRow">
            <h1 className="brand">Rep Coach</h1>
            <span className="liveDot">Live</span>
          </div>
          <p className="subtitle">Track reps, watch your form, and stay in the groove.</p>
        </header>

        <main>
          <PoseCamera />
        </main>
      </div>
    </div>
  );
}

export default App;
