import React, { useState, useEffect } from 'react';
import LiveRootDetectorSimple from './LiveRootDetectorSimple';
import './MainApp.css';

const MainApp: React.FC = () => {
  const [showCamera, setShowCamera] = useState(false);
  const [currentCategory, setCurrentCategory] = useState<'vine' | 'hazelnut'>('vine');
  const [isLoading, setIsLoading] = useState(true);
  const [modalData, setModalData] = useState<{
    isOpen: boolean;
    title: string;
    steps: string[];
  }>({
    isOpen: false,
    title: '',
    steps: []
  });

  // Debug logging
  console.log('MainApp rendered:', { showCamera, currentCategory, isLoading });

  useEffect(() => {
    // Simulate loading time
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  const handleTileClick = (method: string, steps: string[]) => {
    setModalData({
      isOpen: true,
      title: method,
      steps: steps
    });
  };

  const handleStartCamera = () => {
    setModalData({ isOpen: false, title: '', steps: [] });
    setShowCamera(true);
  };

  const handleBackToMain = () => {
    setShowCamera(false);
  };

  const vineMethods = [
    {
      icon: '🔪',
      label: '45° გრადუსიანი დამყნობა',
      sub: 'Whip & Tongue',
      steps: ['მოამზადე 45° ჭრილი.', 'შეახამე კალამი და საძირე.', 'გამაგრე სამაგრით.']
    },
    {
      icon: '🌿',
      label: 'ჭრილზე დამყნობა',
      sub: 'Splice / Cleft',
      steps: ['გააკეთე ღრმა ჭრილი საძირზე.', 'ჩასვი კალამი ჭრილში.', 'შემოახვიე იზოლაციით.']
    },
    {
      icon: '🌱',
      label: 'გვერდითი დამყნობა',
      sub: 'Side Graft',
      steps: ['შექმენი გვერდითი ჭრილი.', 'გადააბა კალამი გვერდულად.', 'დაფიქსირე ლენტით.']
    },
    {
      icon: '🌳',
      label: 'ტოტზე დამყნობა',
      sub: 'On Branch',
      steps: ['არჩევე ჯანსაღი ტოტი.', 'გააკეთე სუფთა ჭრილი.', 'დაამყენე მჭიდროდ.']
    },
  ];

  const hazelnutMethods = [
    {
      icon: '🔪',
      label: 'თხილის ირიბი ჭრა',
      sub: 'Hazelnut Angled Splice',
      steps: ['მოამზადე ირიბი ჭრილი.', 'დაამთხვიე კალამი.', 'დაფიქსირე მჭიდროდ.']
    },
    {
      icon: '🌿',
      label: 'თხილის ჭრილზე დამყნობა',
      sub: 'Hazelnut Splice',
      steps: ['გაამზადე ჭრილი.', 'ჩასვი კალამი.', 'შემოახვიე.']
    },
    {
      icon: '🌱',
      label: 'თხილის გვერდითი დამყნობა',
      sub: 'Hazelnut Side Graft',
      steps: ['გვერდითი ჭრილი.', 'კალმის გადაბმა.', 'ფიქსაცია.']
    },
    {
      icon: '🌳',
      label: 'თხილის ტოტზე დამყნობა',
      sub: 'Hazelnut Branch',
      steps: ['ტოტის არჩევა.', 'სუფთა ჭრილი.', 'დამყნობა.']
    },
  ];

  const currentMethods = currentCategory === 'vine' ? vineMethods : hazelnutMethods;

  if (isLoading) {
    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        backgroundColor: '#1a1a1a',
        color: 'white',
        fontFamily: 'system-ui, Avenir, Helvetica, Arial, sans-serif'
      }}>
        <div style={{ fontSize: '48px', marginBottom: '20px' }}>🌱</div>
        <h1>Loading AI Root Detection...</h1>
        <p>Please wait while the application initializes.</p>
      </div>
    );
  }

  if (showCamera) {
    return (
      <div className="camera-view">
        <div className="camera-header">
          <button className="back-btn" onClick={handleBackToMain}>
            ← უკან
          </button>
          <h2>AI Root Detection</h2>
        </div>
        <LiveRootDetectorSimple />
      </div>
    );
  }

  return (
    <div className="screen">
      <header className="screen-header">
        <div className="glow"></div>
        <h1>ჭკვიანი სამყნობი მოწყობილობა</h1>
      </header>
      
      <main className="screen-body">
        <div className="category-bar" role="tablist" aria-label="Categories">
          <button 
            className={`category-btn ${currentCategory === 'vine' ? 'active' : ''}`}
            onClick={() => setCurrentCategory('vine')}
            role="tab"
            aria-selected={currentCategory === 'vine'}
          >
            ვაზი · Grapevine
          </button>
          <button 
            className={`category-btn ${currentCategory === 'hazelnut' ? 'active' : ''}`}
            onClick={() => setCurrentCategory('hazelnut')}
            role="tab"
            aria-selected={currentCategory === 'hazelnut'}
          >
            თხილი · Hazelnut
          </button>
        </div>
        
        <h2 className="category-title">
          {currentCategory === 'vine' 
            ? 'ვაზის დამყნობის მეთოდები (Grapevine Grafting)'
            : 'თხილის დამყნობის მეთოდები (Hazelnut Grafting)'
          }
        </h2>
        
        <section className="tiles">
          {currentMethods.map((method, index) => (
            <button 
              key={index}
              className="tile"
              onClick={() => handleTileClick(method.label, method.steps)}
            >
              <span className="tile-icon">{method.icon}</span>
              <span className="tile-label">{method.label}</span>
              <span className="tile-sub">{method.sub}</span>
            </button>
          ))}
          
          {/* AI Vision tile - always visible */}
          <button 
            className="tile wide"
            onClick={() => handleTileClick('AI Vision', [
              'მომდევნო ვერსიაში ჩაერთვება ESP32-CAM.',
              'აირჩევა მოდელი და სიხშირე.'
            ])}
          >
            <span className="tile-icon">🤖</span>
            <span className="tile-label">AI Vision</span>
            <span className="tile-sub">Model-Assisted Guidance</span>
          </button>
        </section>
      </main>

      {/* Modal */}
      {modalData.isOpen && (
        <div className="modal" aria-hidden="false">
          <div className="modal-backdrop" onClick={() => setModalData({ isOpen: false, title: '', steps: [] })}></div>
          <div className="modal-dialog" role="dialog" aria-modal="true">
            <header className="modal-header">
              <h2>ინსტრუქცია — {modalData.title}</h2>
              <button 
                className="icon-btn" 
                onClick={() => setModalData({ isOpen: false, title: '', steps: [] })}
                aria-label="დახურვა"
              >
                ✕
              </button>
            </header>
            <div className="modal-content">
              <ul className="steps">
                {modalData.steps.map((step, index) => (
                  <li key={index}>{step}</li>
                ))}
              </ul>
            </div>
            <footer className="modal-footer">
              <button className="cta" onClick={handleStartCamera}>
                დაწყება
              </button>
            </footer>
          </div>
        </div>
      )}
    </div>
  );
};

export default MainApp;
