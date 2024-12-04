import React, { useState } from 'react';
import './App.css';
import Dashboard from './components/Dashboard';
import VideoLibrary from './components/VideoLibrary';
import VideoPlayer from './components/VideoPlayer';
import Banner from './components/Banner';

function App() {
  const [activeTab, setActiveTab] = useState("Dashboard");
  const [watchedVideos, setWatchedVideos] = useState([]); // Trạng thái video đã xem
  const [currentVideo, setCurrentVideo] = useState(null); // Trạng thái video hiện tại

  const handleTabClick = (tab) => {
    setActiveTab(tab);
  };

  const selectVideo = (video) => {
    setCurrentVideo(video); // Cập nhật video hiện tại
    setActiveTab("VideoPlayer"); // Chuyển sang VideoPlayer
  };

  const addWatchedVideo = (video) => {
    setWatchedVideos((prevVideos) => {
      // Tránh thêm trùng lặp video
      if (!prevVideos.find((v) => v.id === video.id)) {
        return [...prevVideos, video];
      }
      return prevVideos;
    });
  };

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <img
            src="/Cam.png"
            alt="AI Camera Logo"
            style={{ width: '150px', height: 'auto' }}
          />
        </div>
        <nav className="nav">
          <button
            className={activeTab === "Dashboard" ? "active" : ""}
            onClick={() => handleTabClick("Dashboard")}
          >
            Dashboard
          </button>
          <button
            className={activeTab === "VideoLibrary" ? "active" : ""}
            onClick={() => handleTabClick("VideoLibrary")}
          >
            Video Library
          </button>
        </nav>
      </header>

      <Banner
        imageUrl="/Cam1.png"
        altText="Welcome to AI Camera"
      />

      <main className="main-content">
        {activeTab === "Dashboard" && <Dashboard watchedVideos={watchedVideos} />}
        {activeTab === "VideoLibrary" && (
          <VideoLibrary
            selectVideo={selectVideo}
          />
        )}
        {activeTab === "VideoPlayer" && currentVideo && (
          <VideoPlayer
            video={currentVideo}
            addWatchedVideo={addWatchedVideo}
          />
        )}
      </main>
    </div>
  );
}

export default App;
