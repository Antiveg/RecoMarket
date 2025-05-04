import React, { useState, useEffect } from "react"
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import ai from '../../assets/ai.svg'

const Evaluation = () => {

    const sideBar = {
        // "AI Performance": <Performance/>,
        // "Documentation": <Documentation/>,
        // "IPYNB Codebase": <Codebase/>,
        // "ML Models Used": <Models/>
        "AI Performance": 1,
        "Documentation": 1,
        "IPYNB Codebase": 1,
        "ML Models Used": 1,
    }

    const [active, setActive] = useState(Object.keys(sideBar)[0])

    return (
        <div className="flex flex-row w-full h-full bg-blue-500 gap-4">
            <aside className="flex flex-col h-full w-1/4 bg-red-500 p-2 gap-2">
                {Object.keys(sideBar).map((sub, index) => (
                    <h2 key={index} className="font-bold text-lg p-2 w-full h-auto bg-white line-clamp-1" onClick={() => setActive(sub)}>{sub}</h2>
                ))}
            </aside>
            <article className="flex flex-col flex-grow bg-yellow-500">
                
            </article>
        </div>
    )
}

export default Evaluation