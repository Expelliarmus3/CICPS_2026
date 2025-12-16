import React, { useState } from 'react';
import { Linkedin, Twitter, Facebook, Instagram, Upload, Menu, X, ArrowRight, User, Lock, Mail, Loader } from 'lucide-react';

const Header = ({ navigateTo }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <nav className="w-full bg-white shadow-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16 items-center">
          {/* Logo Section */}
          <div className="flex-shrink-0 cursor-pointer" onClick={() => navigateTo('home')}>
             <h1 className="text-xl font-bold text-teal-700 tracking-wider">SKIN CHECK</h1>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <button onClick={() => navigateTo('home')} className="text-gray-600 hover:text-teal-600 font-medium transition">Home</button>
            <button className="text-gray-600 hover:text-teal-600 font-medium transition">About</button>
            <button className="text-gray-600 hover:text-teal-600 font-medium transition">Services</button>
            <button className="text-gray-600 hover:text-teal-600 font-medium transition">Contact</button>
            <button 
              onClick={() => navigateTo('diagnosis')}
              className="bg-teal-600 text-white px-5 py-2 rounded-full font-medium hover:bg-teal-700 transition shadow-md"
            >
              SIGN IN
            </button>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="text-gray-600 hover:text-gray-900 focus:outline-none"
            >
              {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {isMenuOpen && (
        <div className="md:hidden bg-white border-t border-gray-100">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            <button onClick={() => {navigateTo('home'); setIsMenuOpen(false)}} className="block w-full text-left px-3 py-2 text-base font-medium text-gray-700 hover:text-teal-600 hover:bg-gray-50 rounded-md">Home</button>
            <button className="block w-full text-left px-3 py-2 text-base font-medium text-gray-700 hover:text-teal-600 hover:bg-gray-50 rounded-md">About</button>
            <button className="block w-full text-left px-3 py-2 text-base font-medium text-gray-700 hover:text-teal-600 hover:bg-gray-50 rounded-md">Services</button>
            <button 
              onClick={() => {navigateTo('diagnosis'); setIsMenuOpen(false)}}
              className="block w-full text-left px-3 py-2 text-base font-medium text-teal-600 font-bold bg-teal-50 rounded-md mt-4"
            >
              SIGN IN
            </button>
          </div>
        </div>
      )}
    </nav>
  );
};

const Footer = () => {
  return (
    <footer className="bg-gray-900 text-gray-300 py-12 border-t border-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          <div className="col-span-2 md:col-span-1">
            <h3 className="text-white text-lg font-bold mb-4">SKIN CHECK</h3>
            <p className="text-sm text-gray-400 mb-4">
              Advanced AI-powered skin disease detection designed for early diagnosis and peace of mind.
            </p>
            <div className="flex space-x-4">
              <a href="#" className="text-gray-400 hover:text-teal-400 transition"><Linkedin size={20} /></a>
              <a href="#" className="text-gray-400 hover:text-teal-400 transition"><Twitter size={20} /></a>
              <a href="#" className="text-gray-400 hover:text-teal-400 transition"><Facebook size={20} /></a>
              <a href="#" className="text-gray-400 hover:text-teal-400 transition"><Instagram size={20} /></a>
            </div>
          </div>
          
          <div>
            <h4 className="text-white font-semibold mb-4 uppercase tracking-wider text-sm">Product</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="#" className="hover:text-white transition">Features</a></li>
              <li><a href="#" className="hover:text-white transition">Technology</a></li>
              <li><a href="#" className="hover:text-white transition">Security</a></li>
              <li><a href="#" className="hover:text-white transition">Pricing</a></li>
            </ul>
          </div>

          <div>
            <h4 className="text-white font-semibold mb-4 uppercase tracking-wider text-sm">Resources</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="#" className="hover:text-white transition">Documentation</a></li>
              <li><a href="#" className="hover:text-white transition">Guides</a></li>
              <li><a href="#" className="hover:text-white transition">API Status</a></li>
              <li><a href="#" className="hover:text-white transition">Help Center</a></li>
            </ul>
          </div>

          <div>
            <h4 className="text-white font-semibold mb-4 uppercase tracking-wider text-sm">Company</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="#" className="hover:text-white transition">About</a></li>
              <li><a href="#" className="hover:text-white transition">Blog</a></li>
              <li><a href="#" className="hover:text-white transition">Jobs</a></li>
              <li><a href="#" className="hover:text-white transition">Legal</a></li>
            </ul>
          </div>
        </div>
        <div className="mt-12 pt-8 border-t border-gray-800 text-center text-xs text-gray-500">
          &copy; 2024 Skin Disease Detector. All rights reserved.
        </div>
      </div>
    </footer>
  );
};

// --- PAGE 1 CONTENT ---
const HomeView = ({ onNavigate }) => {
  return (
    <div className="bg-gray-50 min-h-[calc(100vh-64px)]">
      {/* Hero Header Section */}
      <div className="bg-teal-700 text-white py-16 text-center shadow-lg">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-2 uppercase">
          Skin Disease Detector
        </h1>
        <p className="text-teal-100 text-lg md:text-xl font-light italic">
          "better to be safe than sorry"
        </p>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="flex flex-col lg:flex-row gap-12 items-start">
          
          {/* Left Column: About Section */}
          <div className="flex-1 space-y-6">
            <div className="bg-white p-8 rounded-2xl shadow-sm border border-gray-100">
              <h2 className="text-3xl font-bold text-gray-800 mb-2">About</h2>
              <h3 className="text-lg text-teal-600 font-medium mb-6">
                AI-Driven Dermatological Analysis
              </h3>
              
              <div className="prose text-gray-600 leading-relaxed space-y-4">
                <p>
                  Welcome to our state-of-the-art Skin Disease Detector. This platform utilizes 
                  advanced machine learning algorithms to analyze skin images and provide preliminary 
                  diagnostic insights. Our goal is to make early detection accessible to everyone.
                </p>
                <p>
                  Simply upload a clear image of the affected area, and our system will process 
                  visual patterns to identify potential conditions. While this tool provides 
                  valuable information, it is not a substitute for professional medical advice.
                  Always consult with a certified dermatologist for a definitive diagnosis.
                </p>
                <p>
                  Sign in to access your history, save results, and utilize our advanced 
                  image processing tools. Your health data is encrypted and secure.
                </p>
              </div>

              <div className="mt-8 flex gap-4">
                <button className="text-teal-600 font-semibold flex items-center hover:text-teal-800 transition">
                  Learn more <ArrowRight size={16} className="ml-2" />
                </button>
              </div>
            </div>
          </div>

          {/* Right Column: Login Form */}
          <div className="w-full lg:w-96 flex-shrink-0">
            <div className="bg-white p-8 rounded-2xl shadow-lg border border-gray-100">
              <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-gray-900">Welcome Back</h2>
                <p className="text-gray-500 text-sm mt-2">Please enter your details to sign in</p>
              </div>

              <form className="space-y-5" onSubmit={(e) => { e.preventDefault(); onNavigate('diagnosis'); }}>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Email Address</label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <Mail size={18} className="text-gray-400" />
                    </div>
                    <input 
                      type="email" 
                      placeholder="you@example.com" 
                      className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-teal-500 focus:border-teal-500 transition sm:text-sm"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <Lock size={18} className="text-gray-400" />
                    </div>
                    <input 
                      type="password" 
                      placeholder="••••••••" 
                      className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-teal-500 focus:border-teal-500 transition sm:text-sm"
                    />
                  </div>
                </div>

                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center">
                    <input id="remember-me" type="checkbox" className="h-4 w-4 text-teal-600 focus:ring-teal-500 border-gray-300 rounded" />
                    <label htmlFor="remember-me" className="ml-2 text-gray-600">Remember me</label>
                  </div>
                  <a href="#" className="font-medium text-teal-600 hover:text-teal-500">Forgot password?</a>
                </div>

                <button 
                  type="submit" 
                  className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-teal-600 hover:bg-teal-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500 transition-colors"
                >
                  Sign In
                </button>
              </form>
              
              <div className="mt-6 text-center text-sm text-gray-500">
                Don't have an account? <a href="#" className="font-medium text-teal-600 hover:text-teal-500">Sign up</a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// --- PAGE 2 CONTENT ---
const DiagnosisView = () => {
  const [selectedImage, setSelectedImage] = useState(null); // URL for preview
  const [selectedFile, setSelectedFile] = useState(null);   // Actual file object for upload
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedImage(URL.createObjectURL(file));
      setSelectedFile(file);
      setResult(null);
      setError(null);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImage(URL.createObjectURL(file));
      setSelectedFile(file);
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Connect to the Flask Backend
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Server response was not ok');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Failed to connect to the server. Please ensure the backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-gray-50 min-h-[calc(100vh-64px)]">
      {/* Hero Header Section */}
      <div className="bg-teal-700 text-white py-16 text-center shadow-lg">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-2 uppercase">
          Skin Disease Detector
        </h1>
        <p className="text-teal-100 text-lg md:text-xl font-light italic">
          "better to be safe than sorry"
        </p>
      </div>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
          <div className="p-8">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900 uppercase tracking-wide">Image File</h2>
              <p className="text-gray-500 mt-2">Upload a clear photo of the skin area for analysis</p>
            </div>

            {/* Upload Area */}
            <div 
              className={`relative border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center transition-all duration-200 ease-in-out cursor-pointer group
                ${isDragging ? 'border-teal-500 bg-teal-50' : 'border-gray-300 hover:border-teal-400 hover:bg-gray-50'}
                ${selectedImage ? 'bg-gray-50' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => document.getElementById('file-upload').click()}
            >
              <input 
                id="file-upload" 
                type="file" 
                className="hidden" 
                accept="image/png, image/jpeg, image/jpg"
                onChange={handleFileSelect}
              />
              
              {selectedImage ? (
                <div className="relative w-full max-w-md aspect-video rounded-lg overflow-hidden shadow-md">
                   <img src={selectedImage} alt="Preview" className="w-full h-full object-cover" />
                   <div className="absolute inset-0 bg-black bg-opacity-40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                      <p className="text-white font-medium">Click to change image</p>
                   </div>
                </div>
              ) : (
                <>
                  <div className="w-16 h-16 bg-teal-100 text-teal-600 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                    <Upload size={32} />
                  </div>
                  <p className="text-lg font-medium text-gray-700">Drag & Drop or Click to Upload</p>
                  <p className="text-sm text-gray-400 mt-2">Supported formats: JPEG, JPG, PNG</p>
                </>
              )}
            </div>

            <div className="mt-8 flex justify-center">
              <button 
                onClick={(e) => { e.stopPropagation(); handleSubmit(); }}
                disabled={!selectedImage || isLoading}
                className={`px-8 py-3 rounded-lg font-bold text-lg shadow-md transition-all flex items-center
                  ${selectedImage && !isLoading
                    ? 'bg-teal-600 text-white hover:bg-teal-700 hover:shadow-lg transform hover:-translate-y-0.5' 
                    : 'bg-gray-200 text-gray-400 cursor-not-allowed'}`}
              >
                {isLoading ? (
                  <>
                    <Loader className="animate-spin mr-2" size={20} />
                    Processing...
                  </>
                ) : (
                  "Submit for Analysis"
                )}
              </button>
            </div>
          </div>

          {/* Diagnosis Result Section */}
          <div className="bg-gray-50 p-8 border-t border-gray-100">
             <h3 className="text-xl font-bold text-gray-800 mb-4 uppercase tracking-wide border-b border-gray-200 pb-2">
               Diagnosis Result
             </h3>
             <div className="min-h-[120px] flex items-center justify-center bg-white rounded-lg border border-gray-200 p-6">
                {result ? (
                   <div className="text-center w-full animate-in fade-in duration-500">
                      <p className="text-sm text-gray-500 uppercase font-semibold mb-1">Detected Condition</p>
                      <p className="text-2xl text-teal-800 font-bold mb-2">
                        {result.class}
                      </p>
                      <div className="inline-block bg-teal-100 text-teal-800 text-xs px-2 py-1 rounded-full mb-4">
                        Confidence: {result.confidence}
                      </div>
                      <p className="text-gray-600 text-base max-w-lg mx-auto">
                        {result.description}
                      </p>
                   </div>
                ) : error ? (
                   <div className="text-center text-red-500">
                      <p className="font-bold">Error</p>
                      <p>{error}</p>
                   </div>
                ) : (
                   <p className="text-gray-400 italic">
                     Upload an image and submit to see diagnostic results here.
                   </p>
                )}
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default function App() {
  const [currentPage, setCurrentPage] = useState('home');

  return (
    <div className="min-h-screen flex flex-col font-sans">
      <Header navigateTo={setCurrentPage} />
      
      <main className="flex-grow">
        {currentPage === 'home' ? (
          <HomeView onNavigate={setCurrentPage} />
        ) : (
          <DiagnosisView />
        )}
      </main>

      <Footer />
    </div>
  );
}