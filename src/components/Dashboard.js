import React from 'react';

function Dashboard({ watchedVideos }) {
  return (
    <div className="dashboard">
      <h2>Video đã xem</h2>
      <div className="watched-videos">
        {watchedVideos.length > 0 ? (
          <div className="video-list">
            {watchedVideos.map((video, index) => (
              <div key={index} className="video-item">
                {/* Hiển thị thumbnail của video */}
                <div className="video-thumbnail">
                  <video width="200" height="100" muted>
                    <source src={video.url} type="video/mp4" />
                    Your browser does not support the video tag.
                  </video>
                </div>
                {/* Hiển thị tên video và ngày xem */}
                <h3 className="video-title">{video.title}</h3>
                <p className="video-date">Ngày xem: {video.date}</p>
              </div>
            ))}
          </div>
        ) : (
          <p>Chưa xem video nào.</p>
        )}
      </div>
    </div>
  );
}

export default Dashboard;
