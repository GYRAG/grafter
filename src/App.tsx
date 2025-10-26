import MainApp from './components/MainApp'
import ErrorBoundary from './components/ErrorBoundary'
import './App.css'

// Main App component - AI Root Detection System
function App() {
  return (
    <ErrorBoundary>
      <MainApp />
    </ErrorBoundary>
  )
}

export default App
