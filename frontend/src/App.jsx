import React from 'react';
import { BrowserRouter as Router, Link, Route, Routes } from 'react-router-dom';
// import About from './pages/About/About'
// import Home from './pages/Home/Home'
// import Create from './pages/Create/Create'
// import Login from './pages/Login/Login'
// import Register from './pages/Register/Register'
// import Profile from './pages/Profile/Profile'
import AI from './pages/AI'
import ErrorPage from './pages/ErrorPage'
import Layout from './components/Layout'
import './global.css'

function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<Layout><AI/></Layout>}/>
                <Route path="/ai" element={<Layout><AI/></Layout>}/>
                {/* <Route path="*" element={<ErrorPage 
                    statusNumber={404} 
                    statusMsg={'NotFound'} 
                    addMsg={'Oops, looks like you lost somewhere...'}
                    route={'/login'}
                    routePage={'Login'}/>} /> */}
            </Routes>
        </Router>
    );
}

export default App;
