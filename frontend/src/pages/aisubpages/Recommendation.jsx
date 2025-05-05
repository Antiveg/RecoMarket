import React, { useState, useEffect } from "react"
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import ai from '../../assets/ai.svg'
import next from '../../assets/next.svg'
import ProductCard from "../../components/ProductCard"

const purchasetemps = [
    {
        product_id: 1,
        product_name: 'Hello',
        product_img: ai,
        aisle_id: 1,
        aisle_name: 'Aisle Lorem Ipsum',
        department_id: 1,
        department_name: 'Department Lorem Ipsum',
        totalpurchasecount: 90
    },
    {
        product_id: 2,
        product_name: 'Hello',
        product_img: ai,
        aisle_id: 1,
        aisle_name: 'Aisle Lorem Ipsum',
        department_id: 1,
        department_name: 'Department Lorem Ipsum',
        totalpurchasecount: 100
    },
    {
        product_id: 3,
        product_name: 'Hello',
        product_img: ai,
        aisle_id: 1,
        aisle_name: 'Aisle Lorem Ipsum',
        department_id: 1,
        department_name: 'Department Lorem Ipsum',
        totalpurchasecount: 100
    },
    {
        product_id: 4,
        product_name: 'Hello',
        product_img: ai,
        aisle_id: 1,
        aisle_name: 'Aisle Lorem Ipsum',
        department_id: 1,
        department_name: 'Department Lorem Ipsum',
        totalpurchasecount: 100
    },
    {
        product_id: 5,
        product_name: 'Hello',
        product_img: ai,
        aisle_id: 1,
        aisle_name: 'Aisle Lorem Ipsum',
        department_id: 1,
        department_name: 'Department Lorem Ipsum',
        totalpurchasecount: 100
    },
    {
        product_id: 6,
        product_name: 'Hello',
        product_img: ai,
        aisle_id: 1,
        aisle_name: 'Aisle Lorem Ipsum',
        department_id: 1,
        department_name: 'Department Lorem Ipsum',
        totalpurchasecount: 100
    },
    {
        product_id: 7,
        product_name: 'Hello',
        product_img: ai,
        aisle_id: 1,
        aisle_name: 'Aisle Lorem Ipsum',
        department_id: 1,
        department_name: 'Department Lorem Ipsum',
        totalpurchasecount: 70
    },
    {
        product_id: 8,
        product_name: 'Hello',
        product_img: ai,
        aisle_id: 1,
        aisle_name: 'Aisle Lorem Ipsum',
        department_id: 1,
        department_name: 'Department Lorem Ipsum',
        totalpurchasecount: 100
    },
    {
        product_id: 9,
        product_name: 'Hello',
        product_img: ai,
        aisle_id: 1,
        aisle_name: 'Aisle Lorem Ipsum',
        department_id: 1,
        department_name: 'Department Lorem Ipsum',
        totalpurchasecount: 50
    }
]

const ordertemps = [
    {
        order_id: 1,
        products: [
            {
                product_id: 1,
                product_name: 'Product Lorem',
                product_img: 'XXXXLinks',
                department_id: 1,
                department_name: 'Department Lorem',
                aisle_id: 1,
                aisle_name: 'Aisle Lorem'
            },
            {
                product_id: 1,
                product_name: 'Product Lorem',
                product_img: 'XXXXLinks',
                department_id: 1,
                department_name: 'Department Lorem',
                aisle_id: 1,
                aisle_name: 'Aisle Lorem'
            }
        ]
    },
    {
        order_id: 1,
        products: [
            {
                product_id: 1,
                product_name: 'Product Lorem',
                product_img: 'XXXXLinks',
                department_id: 1,
                department_name: 'Department Lorem',
                aisle_id: 1,
                aisle_name: 'Aisle Lorem'
            },
            {
                product_id: 1,
                product_name: 'Product Lorem',
                product_img: 'XXXXLinks',
                department_id: 1,
                department_name: 'Department Lorem',
                aisle_id: 1,
                aisle_name: 'Aisle Lorem'
            }
        ]
    },
    {
        order_id: 1,
        products: [
            {
                product_id: 1,
                product_name: 'Product Lorem',
                product_img: 'XXXXLinks',
                department_id: 1,
                department_name: 'Department Lorem',
                aisle_id: 1,
                aisle_name: 'Aisle Lorem'
            },
            {
                product_id: 1,
                product_name: 'Product Lorem',
                product_img: 'XXXXLinks',
                department_id: 1,
                department_name: 'Department Lorem',
                aisle_id: 1,
                aisle_name: 'Aisle Lorem'
            }
        ]
    },
    {
        order_id: 1,
        products: [
            {
                product_id: 1,
                product_name: 'Product Lorem',
                product_img: 'XXXXLinks',
                department_id: 1,
                department_name: 'Department Lorem',
                aisle_id: 1,
                aisle_name: 'Aisle Lorem'
            },
            {
                product_id: 1,
                product_name: 'Product Lorem',
                product_img: 'XXXXLinks',
                department_id: 1,
                department_name: 'Department Lorem',
                aisle_id: 1,
                aisle_name: 'Aisle Lorem'
            }
        ]
    }
]

const Recommendation = () => {

    const uploads = import.meta.env.VITE_UPLOADS_BASEURL
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    const [users, setUsers] = useState([])
    const [products, setProducts] = useState([])
    const [orders, setOrders] = useState([])
    const [recommendations, setRecommendations] = useState([])
    
    const [userid, setUserId] = useState(-1)
    const [collabweight, setCollabWeight] = useState(1)
    const [contentweight, setContentWeight] = useState((1 - collabweight).toFixed(2))

    const [currPage, setCurrPage] = useState(0)
    const [nshown, setNShown] = useState(6)
    const [maxPage, setMaxPage] = useState(2)

    const movePagination = (step) => {
        if(currPage + step < 0 || currPage + step >= maxPage){
            return
        }
        setCurrPage(currPage + step)
    }

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true)
                // setProducts(purchasetemps)
                // setOrders(ordertemps)
                const randoms = products
                    .sort(() => Math.random() - 0.5)
                    .slice(0, 5)
                setRecommendations(randoms)

                const backend = import.meta.env.VITE_BACKEND_BASEURL

                const user_response = await axios.get(`${backend}/users`)
                setUsers(user_response.data.users)
                if(user_response.data.users.length > 0 && !userid){
                    setUserId(user_response.data.users[0].user_id)
                }

                const curr_user = (userid != -1 ? userid : user_response.data.users[0].user_id)
                const req_userorders = axios.get(`${backend}/user-orders/${curr_user}`)
                const req_userproducts = axios.get(`${backend}/user-products/${curr_user}`)
                const req_recommendations = axios.post(`${backend}/recommendations/${curr_user}`, {
                    collabweight, contentweight
                })
                const responses = await Promise.all([req_userorders, req_userproducts, req_recommendations])

                setOrders(responses[0].data.userOrders)
                setProducts(responses[1].data.userProducts)
                setRecommendations(responses[2].data.recommendations)
                setNShown(6)
                setMaxPage(Math.ceil(responses[1].data.userProducts.length/6))
            }catch(error){
                setError(error)
            }finally{
                setLoading(false)
            }
        }
        fetchData()
    }, [userid, collabweight])

    const handleUserChange = (e) => {
        const selectedId = e.target.value
        setUserId(selectedId)
    }

    if(loading){
        return (
            <div className='w-full h-full flex justify-center products-center text-base'>
                Fetching Necessary Information for Recommendation from Database ...
            </div>
        )
    }

    if(error !== ''){
        return (
            <div className='w-full h-full flex justify-center products-center text-base'>
                {error}
            </div>
        )
    }

    return (
        <div className="h-full w-full gap-4 pb-8 flex flex-row">
            <div className="h-full w-full flex flex-col gap-4">
                <div className="flex flex-col flex-grow h-full w-full p-2 gap-2 bg-[var(--secondary)] rounded-md">
                    <h2 className="font-bold text-base text-center">Try Recommending Products</h2>
                    <div className="flex flex-row w-auto h-full gap-2 p-2 mt-4 mb-4 m-8 justify-center products-center">
                        {recommendations != null && recommendations.map((rec) => (
                            <div className="flex flex-col h-full p-2 w-full bg-[var(--surface)] rounded-md w-1/5 border-4 border-[var(--primary)]" key={rec.product_id}>
                                <img src={`${uploads}/${rec.department_img}`} alt="no img" className='flex-grow h-0 object-contain'/>
                                <div>
                                    <h3 className='text-center font-bold text-md line-clamp-1'>{rec.product_name}</h3>
                                    <p className='text-center text-sm line-clamp-1'>{rec.department_name} Department</p>
                                    <p className='text-center text-sm line-clamp-1'>{rec.aisle_name}</p>
                                    <p className='font-italic text-right text-xs line-clamp-1 mt-2'>Product ID{rec.product_id}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                    <div className="flex flex-row w-full h-auto gap-4 p-2">
                        <div className="flex flex-col gap-2 w-full">
                            <label htmlFor="userid" className="line-clamp-1">Current User</label>
                            <select className="h-auto p-1 bg-[var(--surface)] rounded-md"
                            name="userid" id="user"
                            onChange={handleUserChange}
                            value={userid}>
                                {users.map((user) => (
                                    <option key={user?.user_id} value={user?.user_id}>{user?.user_id}. {user.user_name}</option>
                                ))}
                            </select>
                        </div>
                        <div className="flex flex-col gap-2 w-full">
                            <label htmlFor="wcollab" className="line-clamp-1">% Collaborative Rec.</label>
                            <select
                                name="wcollab"
                                id="wcollab"
                                className="p-1 bg-gray-200 rounded-md"
                                value={collabweight * 100}
                                onChange={(e) => {
                                    const value = e.target.value;
                                    setCollabWeight((value / 100).toFixed(2))
                                    setContentWeight((1 - value / 100).toFixed(2))
                                }}
                                disabled
                            >
                                {[...Array(11).keys()].map(i => (
                                    <option key={i} value={i * 10}>
                                        {i * 10}
                                    </option>
                                ))}
                            </select>
                        </div>
                        <div className="flex flex-col gap-2 w-full">
                            <label htmlFor="wcontent" className="line-clamp-1">% Content-Based Rec.</label>
                            <select
                                name="wcontent"
                                id="wcontent"
                                className="p-1 bg-gray-200 rounded-md"
                                value={contentweight * 100}
                                onChange={(e) => {
                                    const value = e.target.value
                                    setContentWeight((value / 100).toFixed(2))
                                    setCollabWeight((1 - value / 100).toFixed(2))
                                }}
                                disabled
                            >
                                {[...Array(11).keys()].map(i => (
                                    <option key={i} value={i * 10}>
                                        {i * 10}
                                    </option>
                                ))}
                            </select>
                        </div>
                    </div>
                </div>
                <div className="flex flex-col h-auto p-2 gap-2 bg-[var(--secondary)] rounded-md">
                    <h2 className="font-bold text-base text-center">User Purchases Count</h2>
                    {products &&
                    <div className="flex flex-grow gap-2">
                        <img src={next} alt="no img" className="h-full w-[20px] object-contain" value={-1} onClick={() => movePagination(-1)}/>
                        <div className={`h-28 w-full grid lg:grid-cols-6 md:grid-cols-3 grid-rows-1 gap-2`}>
                            {products != null && products.slice(currPage * nshown, currPage * nshown + nshown).map((item) => (
                                <div className="flex flex-col h-full w-full p-2 w-full bg-[var(--surface)] rounded-sm" key={item.product_id}>
                                    <img src={`${uploads}/${item.department_img}`} alt="no img" className='flex-grow h-0 object-contain'/>
                                    <h3 className='text-sm text-center line-clamp-1'>{item.product_name}</h3>
                                    <p className='font-bold text-center text-sm line-clamp-1'>{item.purchase_count}x</p>
                                </div>
                            ))}
                        </div>
                        <img src={next} alt="no img" className="h-full w-[20px] object-contain transform scale-x-[-1]" onClick={() => movePagination(+1)}/>
                    </div>
                    }
                </div>
            </div>
            <div className="h-full min-w-[160px] max-w-[400px] w-1/3 flex flex-col p-2 gap-2 bg-[var(--secondary)] rounded-md">
                <h1 className="font-bold text-base text-center">User History</h1>
                <div className="h-full w-full flex flex-col overflow-y-auto">
                {(!orders || orders.length <= 0) ? (
                    <p>This user hasn't make any purchase yet...</p>
                ) : (
                    orders.map((order) => (
                        <div key={order?.order_id} className="flex flex-col gap-2 pt-1 pb-1">
                            <h2 className="text-sm">Order Number {order?.order_id}</h2>
                            {order?.orderproducts.map((product) => (
                                <ProductCard product={product}/>
                            ))}
                        </div>
                    ))
                )}
                </div>
            </div>
        </div>
    )
}

export default Recommendation