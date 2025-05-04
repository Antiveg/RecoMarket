import React from "react"
import ErrorPage from './../pages/ErrorPage'
import { Link } from 'react-router-dom'

const Layout = ({ children }) => {

    // const handleLogout = async () => {
    //     try {
    //         localStorage.clear()
    //         await axios.post('http://localhost:5173/auth/logout', {}, {
    //             withCredetials: true
    //         })
    //     }catch(error){
    //         console.log(error)
    //     }
    // }

    // const user = JSON.parse(localStorage.getItem('user'))

    // if(!user){
    //     return (
    //         <ErrorPage
    //         statusNumber={401}
    //         statusMsg={'Unauthorized'}
    //         addMsg="looks like you haven't logged in or you user info lost in localStorage. Please try login again"
    //         route={'/login'}
    //         routePage={'Login'} />
    //     )
    // }

    return (
        <div className="flex flex-col w-screen h-screen bg-[var(--background)] text-[var(--text-primary)]">
            <header className="h-auto">
                <nav className="flex justify-center w-full h-auto p-4 bg-[var(--primary)] text-[var(--text-primary)]">
                    <Link to="/ai" className="text-inherit text-lg font-bold">AI Analysis</Link>
                    {/* <Link to="/home" className={styles.link}>Home</Link>
                    <Link to="/about" className={styles.link}>About</Link>
                    <Link to="/create" className={styles.link}>Create</Link>
                    <Link to="/profile" className={styles.link}>Profile</Link>
                    <Link to="/login" className={styles.link}>Login</Link>
                    <Link to="/register" className={styles.link}>Register</Link>
                    <a href="/login" className={styles.link} onClick={handleLogout}>Logout</a> */}
                </nav>
            </header>
            { React.cloneElement(children) }
        </div>
    )
}

export default Layout